from __future__ import annotations

import glob
import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from typing import Optional

from tqdm import tqdm

from src.cutter import cut_and_crop
from src.models import Highlight, slugify
from src.sampler import extract_frames, iter_chunks, video_duration_sec


@dataclass
class PipelineConfig:
    out_dir: str
    processed_dir: Optional[str]
    threshold: int = 7
    chunk_sec: float = 60.0
    fps: float = 1.0
    keep_source: bool = False
    dry_run: bool = False


# Dedup thresholds. IoU catches partial overlaps within a chunk; gap catches the
# common case where the same moment straddles a chunk boundary and produces two
# back-to-back non-overlapping windows.
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
        # Gap is positive when the two intervals are disjoint; clamp to 0 when overlap.
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


def _output_names(dest_dir: str, chunk_idx: int, highlight: Highlight) -> tuple[str, str]:
    slug = slugify(highlight.title) if highlight.title else f"chunk-{chunk_idx:03d}"
    base = f"{chunk_idx:03d}__score-{highlight.score:02d}__{slug}"
    return (
        os.path.join(dest_dir, base + ".mp4"),
        os.path.join(dest_dir, base + ".json"),
    )


def _write_sidecar(path: str, source_path: str, chunk_start: float, chunk_end: float, highlight: Highlight) -> None:
    payload = {
        "source_path": os.path.abspath(source_path),
        "chunk_start_sec": chunk_start,
        "chunk_end_sec": chunk_end,
        "absolute_start_sec": chunk_start + highlight.start_sec,
        "absolute_end_sec": chunk_start + highlight.end_sec,
        "highlight": highlight.to_dict(),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def process_file(source_path: str, analyzer, cfg: PipelineConfig) -> int:
    """Process a single video file. Returns number of clips saved."""
    duration = video_duration_sec(source_path)
    if duration <= 0:
        print(f"[pipeline] Cannot read {source_path}; skipping.")
        return 0

    print(f"\n[pipeline] {os.path.basename(source_path)}  ({duration:.1f}s)")

    stem = os.path.splitext(os.path.basename(source_path))[0]
    dest_dir = os.path.join(cfg.out_dir, stem)
    os.makedirs(dest_dir, exist_ok=True)

    saved_ranges = _seed_saved_ranges(dest_dir)
    if saved_ranges:
        print(f"[pipeline] resuming with {len(saved_ranges)} prior clip range(s) loaded for dedup")

    saved = 0
    chunks = list(iter_chunks(source_path, chunk_sec=cfg.chunk_sec))
    for idx, (start, end) in enumerate(tqdm(chunks, desc="chunks", leave=False), start=1):
        chunk_duration = end - start
        frames = extract_frames(source_path, start, end, fps=cfg.fps)
        if not frames:
            print(f"  [chunk {idx:03d}] no frames extracted; skipping.")
            continue

        highlight = analyzer.analyze(frames, duration_sec=chunk_duration)
        if highlight is None:
            print(f"  [chunk {idx:03d}] analyzer returned no result.")
            continue

        verdict = (
            f"score={highlight.score} highlight={highlight.is_highlight} "
            f"window={highlight.start_sec:.1f}-{highlight.end_sec:.1f}s "
            f"title={highlight.title!r}"
        )

        if not (highlight.is_highlight and highlight.score >= cfg.threshold):
            print(f"  [chunk {idx:03d}] skip  {verdict}")
            continue

        abs_start = start + highlight.start_sec
        abs_end = start + highlight.end_sec

        if _is_duplicate((abs_start, abs_end), saved_ranges):
            print(
                f"  [chunk {idx:03d}] duplicate of prior clip; "
                f"skipping {abs_start:.1f}-{abs_end:.1f}s"
            )
            continue

        if cfg.dry_run:
            print(f"  [chunk {idx:03d}] DRY  {verdict}")
            saved_ranges.append((abs_start, abs_end))
            continue

        # Resume guard: a prior run already produced a clip for this chunk index.
        # Glob by index prefix so a different title-slug doesn't bypass the check.
        existing = glob.glob(os.path.join(dest_dir, f"{idx:03d}__*.mp4"))
        if existing:
            print(f"  [chunk {idx:03d}] exists; skipping {existing[0]}")
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
            print(f"  [chunk {idx:03d}] ffmpeg failed; skipping. stderr tail:\n{stderr}")
            continue

        _write_sidecar(meta_path, source_path, start, end, highlight)
        saved_ranges.append((abs_start, abs_end))
        saved += 1
        print(f"  [chunk {idx:03d}] SAVE {verdict} -> {clip_path}")

    if not cfg.keep_source and not cfg.dry_run and cfg.processed_dir:
        os.makedirs(cfg.processed_dir, exist_ok=True)
        dest = os.path.join(cfg.processed_dir, os.path.basename(source_path))
        shutil.move(source_path, dest)
        print(f"[pipeline] archived -> {dest}")

    return saved
