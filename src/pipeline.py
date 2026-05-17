from __future__ import annotations

import glob
import json
import logging
import os
import shutil
import subprocess
from dataclasses import dataclass
from typing import Optional

from tqdm import tqdm

from src.cutter import cut_and_crop
from src.models import Highlight, ScoutRegion, slugify
from src.sampler import extract_frames, video_duration_sec
from src.scout import iter_scout_windows, merge_regions, split_long_regions

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    out_dir: str
    processed_dir: Optional[str]
    threshold: int = 7
    scout_window_sec: float = 300.0
    scout_overlap_sec: float = 60.0
    scout_fps: float = 0.5
    scout_frame_pixels: int = 240 * 432
    detail_fps: float = 2.0
    detail_frame_pixels: int = 480 * 854
    max_region_sec: float = 90.0
    region_pad_sec: float = 2.0
    keep_source: bool = False
    dry_run: bool = False


# Dedup thresholds. IoU catches partial overlaps; gap catches the case where two
# back-to-back non-overlapping windows describe the same moment.
_DEDUP_IOU = 0.3
_DEDUP_GAP_SEC = 2.0


def _is_duplicate(
    candidate: tuple[float, float], saved: list[tuple[float, float]]
) -> bool:
    cs, ce = candidate
    for ss, se in saved:
        inter = max(0.0, min(ce, se) - max(cs, ss))
        union = max(ce, se) - min(cs, ss)
        iou = inter / union if union > 0 else 0.0
        if iou > _DEDUP_IOU:
            return True
        gap = max(0.0, max(cs, ss) - min(ce, se))
        if iou == 0.0 and gap < _DEDUP_GAP_SEC:
            return True
    return False


def _seed_saved_ranges(dest_dir: str) -> list[tuple[float, float]]:
    ranges: list[tuple[float, float]] = []
    for sc in glob.glob(os.path.join(dest_dir, "*.json")):
        try:
            with open(sc, "r", encoding="utf-8") as f:
                payload = json.load(f)
            a = float(payload["absolute_start_sec"])
            b = float(payload["absolute_end_sec"])
            ranges.append((a, b))
        except (OSError, KeyError, ValueError, json.JSONDecodeError):
            continue
    return ranges


def _output_names(dest_dir: str, region_idx: int, highlight: Highlight) -> tuple[str, str]:
    slug = slugify(highlight.title) if highlight.title else f"region-{region_idx:03d}"
    base = f"{region_idx:03d}__score-{highlight.score:02d}__{slug}"
    return (
        os.path.join(dest_dir, base + ".mp4"),
        os.path.join(dest_dir, base + ".json"),
    )


def _write_sidecar(
    path: str,
    source_path: str,
    region: ScoutRegion,
    abs_start: float,
    abs_end: float,
    highlight: Highlight,
) -> None:
    payload = {
        "source_path": os.path.abspath(source_path),
        "region_start_sec": region.start_sec,
        "region_end_sec": region.end_sec,
        "region_type": region.type,
        "absolute_start_sec": abs_start,
        "absolute_end_sec": abs_end,
        "highlight": highlight.to_dict(),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _run_scout(source_path: str, duration: float, analyzer, cfg: PipelineConfig) -> list[ScoutRegion]:
    windows = list(
        iter_scout_windows(
            duration,
            window_sec=cfg.scout_window_sec,
            overlap_sec=cfg.scout_overlap_sec,
        )
    )
    logger.info("%d scout window(s) over %.1fs", len(windows), duration)

    all_regions: list[ScoutRegion] = []
    for w_idx, (w_start, w_end) in enumerate(
        tqdm(windows, desc="scout", leave=False), start=1
    ):
        w_duration = w_end - w_start
        frames = extract_frames(source_path, w_start, w_end, fps=cfg.scout_fps)
        logger.debug(
            "scout window %d range=%.1f-%.1fs (%.1fs) extracted %d frame(s) at %.2f fps",
            w_idx, w_start, w_end, w_duration, len(frames), cfg.scout_fps,
        )
        if not frames:
            logger.warning("scout window %d: no frames extracted; skipping", w_idx)
            continue
        regions = analyzer.scout(
            frames,
            window_duration_sec=w_duration,
            fps_used=cfg.scout_fps,
            frame_pixels_budget=cfg.scout_frame_pixels,
        )
        # Offset window-relative regions to absolute source time.
        for r in regions:
            all_regions.append(
                ScoutRegion(
                    start_sec=w_start + r.start_sec,
                    end_sec=w_start + r.end_sec,
                    type=r.type,
                )
            )
        logger.info(
            "scout window %d: %.1f-%.1fs -> %d region(s)",
            w_idx, w_start, w_end, len(regions),
        )
        for r in regions:
            logger.debug(
                "scout window %d region: %.1f-%.1fs (abs %.1f-%.1f) type=%s",
                w_idx, r.start_sec, r.end_sec, w_start + r.start_sec, w_start + r.end_sec, r.type,
            )

    return all_regions


def process_file(source_path: str, analyzer, cfg: PipelineConfig) -> int:
    """Process a single video file. Returns number of clips saved."""
    duration = video_duration_sec(source_path)
    if duration <= 0:
        logger.warning("cannot read %s; skipping", source_path)
        return 0

    logger.info("%s  (%.1fs)", os.path.basename(source_path), duration)

    stem = os.path.splitext(os.path.basename(source_path))[0]
    dest_dir = os.path.join(cfg.out_dir, stem)
    os.makedirs(dest_dir, exist_ok=True)
    logger.debug("output dir: %s", dest_dir)

    # Stage 1: scout
    raw_regions = _run_scout(source_path, duration, analyzer, cfg)
    if not raw_regions:
        logger.info("scout found no regions; nothing to process")
        _maybe_archive(source_path, cfg)
        return 0

    merged = merge_regions(raw_regions)
    split = split_long_regions(merged, max_sec=cfg.max_region_sec)
    logger.info(
        "regions: %d raw -> %d merged -> %d after long-split",
        len(raw_regions), len(merged), len(split),
    )
    for i, r in enumerate(split, start=1):
        logger.debug("region %03d: %.1f-%.1fs type=%s", i, r.start_sec, r.end_sec, r.type)

    saved_ranges = _seed_saved_ranges(dest_dir)
    if saved_ranges:
        logger.info("resume: %d prior clip range(s) loaded for dedup", len(saved_ranges))

    # Stage 2: detail
    saved = 0
    for idx, region in enumerate(tqdm(split, desc="detail", leave=False), start=1):
        padded_start = max(0.0, region.start_sec - cfg.region_pad_sec)
        padded_end = min(duration, region.end_sec + cfg.region_pad_sec)
        padded_duration = padded_end - padded_start
        logger.debug(
            "region %03d: type=%s raw=%.1f-%.1fs padded=%.1f-%.1fs (%.1fs)",
            idx, region.type, region.start_sec, region.end_sec,
            padded_start, padded_end, padded_duration,
        )
        if padded_duration <= 0:
            logger.warning("region %03d: degenerate after padding; skipping", idx)
            continue

        frames = extract_frames(source_path, padded_start, padded_end, fps=cfg.detail_fps)
        logger.debug(
            "region %03d: extracted %d frame(s) at %.2f fps", idx, len(frames), cfg.detail_fps,
        )
        if not frames:
            logger.warning("region %03d: no frames; skipping", idx)
            continue

        highlight = analyzer.analyze(
            frames,
            duration_sec=padded_duration,
            fps_used=cfg.detail_fps,
            frame_pixels_budget=cfg.detail_frame_pixels,
            region_type=region.type,
        )
        if highlight is None:
            logger.warning("region %03d: analyzer returned no result", idx)
            continue

        verdict = (
            f"type={region.type} score={highlight.score} "
            f"is_highlight={highlight.is_highlight} "
            f"window={highlight.start_sec:.1f}-{highlight.end_sec:.1f}s "
            f"title={highlight.title!r}"
        )

        if not (highlight.is_highlight and highlight.score >= cfg.threshold):
            logger.info("region %03d: skip  %s", idx, verdict)
            continue

        abs_start = padded_start + highlight.start_sec
        abs_end = padded_start + highlight.end_sec

        if _is_duplicate((abs_start, abs_end), saved_ranges):
            logger.info(
                "region %03d: duplicate of prior clip; skipping %.1f-%.1fs",
                idx, abs_start, abs_end,
            )
            continue

        if cfg.dry_run:
            logger.info("region %03d: DRY  %s", idx, verdict)
            saved_ranges.append((abs_start, abs_end))
            continue

        existing = glob.glob(os.path.join(dest_dir, f"{idx:03d}__*.mp4"))
        if existing:
            logger.info("region %03d: exists; skipping %s", idx, existing[0])
            saved_ranges.append((abs_start, abs_end))
            continue

        clip_path, meta_path = _output_names(dest_dir, idx, highlight)
        try:
            cut_and_crop(
                src=source_path,
                out=clip_path,
                start_sec=abs_start,
                end_sec=abs_end,
                center_x_pct=highlight.action_center_x,
            )
        except subprocess.CalledProcessError as e:
            stderr = (e.stderr or b"").decode("utf-8", errors="replace")[-400:]
            logger.error("region %03d: ffmpeg failed; skipping. stderr tail:\n%s", idx, stderr)
            continue

        _write_sidecar(meta_path, source_path, region, abs_start, abs_end, highlight)
        saved_ranges.append((abs_start, abs_end))
        saved += 1
        logger.info("region %03d: SAVE %s -> %s", idx, verdict, clip_path)

    _maybe_archive(source_path, cfg)
    return saved


def _maybe_archive(source_path: str, cfg: PipelineConfig) -> None:
    if cfg.keep_source or cfg.dry_run or not cfg.processed_dir:
        return
    os.makedirs(cfg.processed_dir, exist_ok=True)
    dest = os.path.join(cfg.processed_dir, os.path.basename(source_path))
    shutil.move(source_path, dest)
    logger.info("archived -> %s", dest)
