from __future__ import annotations

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


def _output_paths(out_root: str, source_path: str, chunk_idx: int, highlight: Highlight) -> tuple[str, str]:
    stem = os.path.splitext(os.path.basename(source_path))[0]
    dest_dir = os.path.join(out_root, stem)
    os.makedirs(dest_dir, exist_ok=True)
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

        if cfg.dry_run:
            print(f"  [chunk {idx:03d}] DRY  {verdict}")
            continue

        clip_path, meta_path = _output_paths(cfg.out_dir, source_path, idx, highlight)
        if os.path.exists(clip_path):
            print(f"  [chunk {idx:03d}] exists; skipping {clip_path}")
            continue

        abs_start = start + highlight.start_sec
        abs_end = start + highlight.end_sec
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
        saved += 1
        print(f"  [chunk {idx:03d}] SAVE {verdict} -> {clip_path}")

    if not cfg.keep_source and not cfg.dry_run and cfg.processed_dir:
        os.makedirs(cfg.processed_dir, exist_ok=True)
        dest = os.path.join(cfg.processed_dir, os.path.basename(source_path))
        shutil.move(source_path, dest)
        print(f"[pipeline] archived -> {dest}")

    return saved
