"""Build a highlight reel from the sidecars in a source's output directory.

Two modes:
  * 9:16 vertical (default) — lossless concat of the existing 9:16 clips. Fast.
  * 16:9 horizontal — re-cut each clip's absolute time range from the source,
    then concat. Slower (re-encodes) but matches FIFA-style match-recap aspect.

The result lands at <dest_dir>/_reel.mp4 (underscore prefix sorts to top).
"""

from __future__ import annotations

import glob
import json
import logging
import os
import subprocess
import tempfile
from dataclasses import dataclass

from src.cutter import has_nvenc

logger = logging.getLogger(__name__)


REEL_FILENAME = "_reel.mp4"
MIN_CLIPS_FOR_REEL = 2


@dataclass
class _SidecarEntry:
    sidecar_path: str
    clip_path: str
    absolute_start_sec: float
    absolute_end_sec: float
    score: int
    region_type: str


def _load_sidecars(dest_dir: str) -> list[_SidecarEntry]:
    """Collect well-formed sidecars in dest_dir, paired with their .mp4."""
    entries: list[_SidecarEntry] = []
    for sc in sorted(glob.glob(os.path.join(dest_dir, "*.json"))):
        try:
            with open(sc, "r", encoding="utf-8") as f:
                payload = json.load(f)
            abs_start = float(payload["absolute_start_sec"])
            abs_end = float(payload["absolute_end_sec"])
            highlight = payload.get("highlight") or {}
            score = int(highlight.get("score", 0))
            region_type = str(payload.get("region_type", "other"))
        except (OSError, KeyError, ValueError, TypeError, json.JSONDecodeError) as e:
            logger.debug("reel: skipping malformed sidecar %s: %s", sc, e)
            continue
        clip = os.path.splitext(sc)[0] + ".mp4"
        if not os.path.exists(clip):
            logger.debug("reel: sidecar %s has no matching mp4; skipping", sc)
            continue
        entries.append(
            _SidecarEntry(
                sidecar_path=sc,
                clip_path=clip,
                absolute_start_sec=abs_start,
                absolute_end_sec=abs_end,
                score=score,
                region_type=region_type,
            )
        )
    return entries


def _write_concat_list(paths: list[str], list_path: str) -> None:
    """Write ffmpeg concat-demuxer list. Paths must not contain single quotes."""
    with open(list_path, "w", encoding="utf-8") as f:
        for p in paths:
            # concat demuxer wants forward slashes and single-quoted paths.
            abs_p = os.path.abspath(p).replace("'", r"'\''")
            f.write(f"file '{abs_p}'\n")


def _concat_lossless(clip_paths: list[str], out_path: str) -> None:
    """Concatenate clips with `-c copy` (no re-encode). Assumes codecs match."""
    with tempfile.TemporaryDirectory() as td:
        list_path = os.path.join(td, "list.txt")
        _write_concat_list(clip_paths, list_path)
        cmd = [
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", list_path,
            "-c", "copy",
            "-movflags", "+faststart",
            out_path,
        ]
        logger.debug("reel concat cmd: %s", " ".join(cmd))
        subprocess.run(cmd, check=True, capture_output=True)


def _recut_segment(
    src: str, out: str, start_sec: float, end_sec: float, use_nvenc: bool
) -> None:
    """Re-encode a [start, end] segment from src to out, preserving source aspect."""
    video_args = (
        ["-c:v", "h264_nvenc", "-preset", "p4", "-cq", "23"]
        if use_nvenc
        else ["-c:v", "libx264", "-preset", "medium", "-crf", "20"]
    )
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{start_sec:.3f}",
        "-to", f"{end_sec:.3f}",
        "-i", src,
        *video_args,
        "-c:a", "aac", "-b:a", "128k",
        "-movflags", "+faststart",
        out,
    ]
    logger.debug("reel segment cmd: %s", " ".join(cmd))
    subprocess.run(cmd, check=True, capture_output=True)


def _build_landscape(
    source_path: str, entries: list[_SidecarEntry], out_path: str
) -> None:
    """Re-cut each entry from source at source aspect, then concat."""
    use_nvenc = has_nvenc()
    with tempfile.TemporaryDirectory() as td:
        seg_paths: list[str] = []
        for i, e in enumerate(entries):
            seg = os.path.join(td, f"seg_{i:03d}.mp4")
            _recut_segment(
                src=source_path,
                out=seg,
                start_sec=e.absolute_start_sec,
                end_sec=e.absolute_end_sec,
                use_nvenc=use_nvenc,
            )
            seg_paths.append(seg)
        # All segments share encoder settings, so concat -c copy is safe.
        list_path = os.path.join(td, "list.txt")
        _write_concat_list(seg_paths, list_path)
        cmd = [
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", list_path,
            "-c", "copy",
            "-movflags", "+faststart",
            out_path,
        ]
        logger.debug("reel concat cmd: %s", " ".join(cmd))
        subprocess.run(cmd, check=True, capture_output=True)


def build_reel(
    dest_dir: str,
    source_path: str,
    landscape: bool = False,
    out_filename: str = REEL_FILENAME,
) -> str | None:
    """Build a highlight reel for the source whose clips live in dest_dir.

    Returns the path to the reel, or None if no reel was built (too few clips).
    Raises subprocess.CalledProcessError on ffmpeg failure.
    """
    entries = _load_sidecars(dest_dir)
    if len(entries) < MIN_CLIPS_FOR_REEL:
        logger.info(
            "reel: only %d clip(s) in %s; need %d — skipping",
            len(entries), dest_dir, MIN_CLIPS_FOR_REEL,
        )
        return None

    entries.sort(key=lambda e: e.absolute_start_sec)
    out_path = os.path.join(dest_dir, out_filename)

    aspect = "16:9 landscape" if landscape else "9:16 vertical"
    logger.info(
        "reel: building %s from %d clip(s) -> %s",
        aspect, len(entries), out_path,
    )
    for i, e in enumerate(entries, start=1):
        logger.debug(
            "reel clip %02d: %.1f-%.1fs type=%s score=%d  %s",
            i, e.absolute_start_sec, e.absolute_end_sec,
            e.region_type, e.score, os.path.basename(e.clip_path),
        )

    if landscape:
        _build_landscape(source_path, entries, out_path)
    else:
        _concat_lossless([e.clip_path for e in entries], out_path)

    return out_path
