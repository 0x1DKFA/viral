from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Optional

from dotenv import load_dotenv

from src.log import setup_logging
from src.pipeline import PipelineConfig, process_file

logger = logging.getLogger(__name__)


def _collect_inputs(target: Optional[str], default_dir: str) -> list[str]:
    if target is None:
        target = default_dir
    if not os.path.exists(target):
        logger.warning("not found: %s", target)
        return []
    if os.path.isfile(target):
        return [target] if target.lower().endswith(".mp4") else []
    return sorted(
        os.path.join(target, f)
        for f in os.listdir(target)
        if f.lower().endswith(".mp4")
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="viral",
        description="Scout + detail VLM pipeline that cuts 9:16 short clips from gameplay video.",
    )
    p.add_argument("path", nargs="?", help="Video file or directory (default: $RAW_RECORDINGS_DIR)")
    p.add_argument("--out", help="Output directory (default: $OUTPUT_DIR or ./output)")
    p.add_argument("--threshold", type=int, help="Minimum viral score 1-10 (default: $VIRAL_THRESHOLD or 7)")
    p.add_argument("--model", help="Hugging Face model id (default: $VLM_MODEL_ID)")

    # Scout pass
    p.add_argument("--scout-window", type=float, default=300.0,
                   help="Scout window length in seconds (default: 300)")
    p.add_argument("--scout-overlap", type=float, default=60.0,
                   help="Scout window overlap in seconds (default: 60)")
    p.add_argument("--scout-fps", type=float, default=0.5,
                   help="Scout frame sampling fps (default: 0.5)")
    p.add_argument("--scout-frame-pixels", type=int, default=240 * 432,
                   help="Scout per-frame pixel budget (default ~240p)")

    # Detail pass
    p.add_argument("--detail-fps", type=float, default=2.0,
                   help="Detail-pass frame sampling fps (default: 2.0)")
    p.add_argument("--detail-frame-pixels", type=int, default=480 * 854,
                   help="Detail per-frame pixel budget (default ~480p)")

    # Region shaping
    p.add_argument("--max-region", type=float, default=90.0,
                   help="Split scout regions longer than this many seconds (default: 90)")
    p.add_argument("--region-pad", type=float, default=2.0,
                   help="Pad each scout region by this many seconds on each side before detail (default: 2)")

    # Clip shaping
    p.add_argument("--min-clip", type=float, default=8.0,
                   help="Minimum saved clip length in seconds (default: 8)")
    p.add_argument("--max-clip", type=float, default=25.0,
                   help="Maximum saved clip length in seconds (default: 25)")
    p.add_argument(
        "--explain-sizing",
        action="store_true",
        help="Ask the detail pass to include recommended_duration_sec + sizing_reason in each sidecar.",
    )

    p.add_argument("--keep-source", action="store_true", help="Do not move processed source files")
    p.add_argument("--dry-run", action="store_true", help="Score regions and print, do not cut clips")

    # Reel
    p.add_argument(
        "--reel-landscape",
        action="store_true",
        help="Build the highlight reel at 16:9 (re-cut from source). Default is 9:16 lossless concat of existing clips.",
    )
    p.add_argument(
        "--no-reel",
        action="store_true",
        help="Skip building the highlight reel.",
    )

    # Logging
    p.add_argument("--log-dir", default=None,
                   help="Per-run log file directory (default: $LOG_DIR or ./logs)")
    p.add_argument("--log-level", default=None,
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                   help="Log verbosity (default: INFO; overridden to DEBUG by --debug)")
    p.add_argument("--debug", action="store_true",
                   help="Shorthand for --log-level DEBUG; also lets noisy libraries log at INFO+")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    load_dotenv()
    args = build_parser().parse_args(argv)

    log_dir = args.log_dir or os.getenv("LOG_DIR", "./logs")
    if args.debug:
        log_level = "DEBUG"
    else:
        log_level = args.log_level or os.getenv("LOG_LEVEL", "INFO")
    log_path = setup_logging(log_dir=log_dir, level=log_level)
    logger.info("log -> %s  (level=%s)", log_path, log_level)

    raw_default = os.getenv("RAW_RECORDINGS_DIR", "./raw_recordings")
    out_dir = args.out or os.getenv("OUTPUT_DIR", "./output")
    processed_dir = os.getenv("PROCESSED_DIR", "./processed")
    threshold = args.threshold if args.threshold is not None else int(os.getenv("VIRAL_THRESHOLD", "7"))
    model_id = args.model or os.getenv("VLM_MODEL_ID", "Qwen/Qwen3-VL-8B-Instruct")

    os.makedirs(out_dir, exist_ok=True)

    inputs = _collect_inputs(args.path, raw_default)
    if not inputs:
        logger.warning("no .mp4 files to process (looked at %s)", args.path or raw_default)
        return 0

    logger.info("%d file(s) to process; threshold=%d model=%s", len(inputs), threshold, model_id)

    # Lazy import to avoid pulling torch/transformers in dry runs that don't reach analysis.
    from src.analyzer import VLMAnalyzer

    analyzer = VLMAnalyzer(
        model_id=model_id,
        min_clip_sec=args.min_clip,
        max_clip_sec=args.max_clip,
        explain_sizing=args.explain_sizing,
    )

    cfg = PipelineConfig(
        out_dir=out_dir,
        processed_dir=processed_dir,
        threshold=threshold,
        scout_window_sec=args.scout_window,
        scout_overlap_sec=args.scout_overlap,
        scout_fps=args.scout_fps,
        scout_frame_pixels=args.scout_frame_pixels,
        detail_fps=args.detail_fps,
        detail_frame_pixels=args.detail_frame_pixels,
        max_region_sec=args.max_region,
        region_pad_sec=args.region_pad,
        keep_source=args.keep_source,
        dry_run=args.dry_run,
        build_reel=not args.no_reel,
        reel_landscape=args.reel_landscape,
    )
    logger.debug("pipeline config: %r", cfg)

    total = 0
    for path in inputs:
        total += process_file(path, analyzer, cfg)

    logger.info("done. saved %d clip(s).", total)
    return 0


if __name__ == "__main__":
    sys.exit(main())
