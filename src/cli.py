from __future__ import annotations

import argparse
import os
import sys
from typing import Optional

from dotenv import load_dotenv

from src.pipeline import PipelineConfig, process_file


def _collect_inputs(target: Optional[str], default_dir: str) -> list[str]:
    if target is None:
        target = default_dir
    if not os.path.exists(target):
        print(f"[cli] not found: {target}")
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
        description="Score gameplay videos with Qwen3-VL and cut 9:16 short clips.",
    )
    p.add_argument("path", nargs="?", help="Video file or directory (default: $RAW_RECORDINGS_DIR)")
    p.add_argument("--out", help="Output directory (default: $OUTPUT_DIR or ./output)")
    p.add_argument("--threshold", type=int, help="Minimum viral score 1-10 (default: $VIRAL_THRESHOLD or 7)")
    p.add_argument("--chunk", type=float, default=60.0, help="Chunk length in seconds (default: 60)")
    p.add_argument("--fps", type=float, default=1.0, help="Frame sampling fps within a chunk (default: 1.0)")
    p.add_argument("--model", help="Hugging Face model id (default: $VLM_MODEL_ID)")
    p.add_argument(
        "--max-frame-pixels",
        type=int,
        default=480 * 854,
        help="Downscale each VLM frame so width*height <= this (default ~480p). Lower = less RAM/VRAM.",
    )
    p.add_argument("--keep-source", action="store_true", help="Do not move processed source files")
    p.add_argument("--dry-run", action="store_true", help="Score chunks and print, do not cut clips")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    load_dotenv()
    args = build_parser().parse_args(argv)

    raw_default = os.getenv("RAW_RECORDINGS_DIR", "./raw_recordings")
    out_dir = args.out or os.getenv("OUTPUT_DIR", "./output")
    processed_dir = os.getenv("PROCESSED_DIR", "./processed")
    threshold = args.threshold if args.threshold is not None else int(os.getenv("VIRAL_THRESHOLD", "7"))
    model_id = args.model or os.getenv("VLM_MODEL_ID", "Qwen/Qwen3-VL-8B-Instruct")

    os.makedirs(out_dir, exist_ok=True)

    inputs = _collect_inputs(args.path, raw_default)
    if not inputs:
        print(f"[cli] no .mp4 files to process (looked at {args.path or raw_default})")
        return 0

    print(f"[cli] {len(inputs)} file(s) to process; threshold={threshold} model={model_id}")

    # Lazy import to avoid pulling torch/transformers in dry runs that don't reach analysis.
    from src.analyzer import VLMAnalyzer

    analyzer = VLMAnalyzer(model_id=model_id, max_frame_pixels=args.max_frame_pixels)

    cfg = PipelineConfig(
        out_dir=out_dir,
        processed_dir=processed_dir,
        threshold=threshold,
        chunk_sec=args.chunk,
        fps=args.fps,
        keep_source=args.keep_source,
        dry_run=args.dry_run,
    )

    total = 0
    for path in inputs:
        total += process_file(path, analyzer, cfg)

    print(f"\n[cli] done. saved {total} clip(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
