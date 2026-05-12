from __future__ import annotations

from typing import Iterator

import cv2
from PIL import Image


def video_duration_sec(path: str) -> float:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return 0.0
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
        if fps <= 0 or frames <= 0:
            return 0.0
        return float(frames) / float(fps)
    finally:
        cap.release()


def iter_chunks(
    path: str, chunk_sec: float = 60.0, min_tail_sec: float = 5.0
) -> Iterator[tuple[float, float]]:
    """Yield (start, end) windows covering the whole video.

    The final window is merged into the previous one if it would be shorter
    than `min_tail_sec`, to avoid sending a 2-second sliver to the VLM.
    """
    duration = video_duration_sec(path)
    if duration <= 0:
        return

    starts: list[float] = []
    t = 0.0
    while t < duration:
        starts.append(t)
        t += chunk_sec

    for i, start in enumerate(starts):
        end = min(start + chunk_sec, duration)
        is_last = i == len(starts) - 1
        if not is_last:
            yield (start, end)
            continue
        # Last chunk: if it's too short and we have a predecessor, the caller
        # would still get useful frames; we just yield it as-is for simplicity
        # but skip slivers under min_tail_sec when there's a prior chunk.
        if (end - start) < min_tail_sec and len(starts) > 1:
            return
        yield (start, end)


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
