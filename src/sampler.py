from __future__ import annotations

import logging

import cv2
from PIL import Image

logger = logging.getLogger(__name__)


def video_duration_sec(path: str) -> float:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        logger.debug("video_duration_sec: cannot open %s", path)
        return 0.0
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
        if fps <= 0 or frames <= 0:
            logger.debug("video_duration_sec: bad probe fps=%.2f frames=%.0f for %s", fps, frames, path)
            return 0.0
        duration = float(frames) / float(fps)
        logger.debug("probe %s: fps=%.2f frames=%.0f duration=%.2fs", path, fps, frames, duration)
        return duration
    finally:
        cap.release()


def extract_frames(
    path: str, start_sec: float, end_sec: float, fps: float = 1.0
) -> list[Image.Image]:
    """Extract frames from [start_sec, end_sec) at the requested sampling fps.

    Returns PIL RGB images. Empty list if the file can't be read.
    """
    if end_sec <= start_sec or fps <= 0:
        return []

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return []

    try:
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        if src_fps <= 0:
            src_fps = 30.0

        step = max(1, int(round(src_fps / fps)))
        start_frame = int(round(start_sec * src_fps))
        end_frame = int(round(end_sec * src_fps))

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        frames: list[Image.Image] = []
        idx = start_frame
        while idx < end_frame:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            if (idx - start_frame) % step == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(rgb))
            idx += 1
        return frames
    finally:
        cap.release()
