from __future__ import annotations

import logging
import shutil
import subprocess

logger = logging.getLogger(__name__)


def _has_nvenc() -> bool:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return False
    try:
        out = subprocess.run(
            [ffmpeg, "-hide_banner", "-encoders"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except subprocess.CalledProcessError:
        return False
    return "h264_nvenc" in out


_NVENC_CACHED: bool | None = None


def has_nvenc() -> bool:
    global _NVENC_CACHED
    if _NVENC_CACHED is None:
        _NVENC_CACHED = _has_nvenc()
    return _NVENC_CACHED


def cut_and_crop(
    src: str,
    out: str,
    start_sec: float,
    end_sec: float,
    center_x_pct: float = 0.5,
    use_nvenc: bool | None = None,
) -> None:
    """Cut [start_sec, end_sec) from `src` and crop to 9:16 around `center_x_pct`.

    Uses h264_nvenc when available; falls back to libx264.
    Raises subprocess.CalledProcessError on ffmpeg failure.
    """
    if end_sec <= start_sec:
        raise ValueError(f"end_sec ({end_sec}) must be > start_sec ({start_sec})")

    cx = max(0.0, min(1.0, float(center_x_pct)))
    crop_filter = f"crop=ih*9/16:ih:iw*{cx}-ow/2:0"

    if use_nvenc is None:
        use_nvenc = has_nvenc()

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
        "-vf", crop_filter,
        *video_args,
        "-c:a", "aac", "-b:a", "128k",
        "-movflags", "+faststart",
        out,
    ]
    logger.debug("ffmpeg cmd: %s", " ".join(cmd))
    subprocess.run(cmd, check=True, capture_output=True)
