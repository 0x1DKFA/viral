import os
import shutil
import subprocess

import pytest

from src.sampler import extract_frames, iter_chunks, video_duration_sec


@pytest.fixture(scope="module")
def synthetic_video(tmp_path_factory):
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not available")
    path = tmp_path_factory.mktemp("vids") / "synthetic.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y", "-f", "lavfi",
            "-i", "testsrc=duration=10:size=320x240:rate=30",
            "-pix_fmt", "yuv420p", str(path),
        ],
        check=True, capture_output=True,
    )
    return str(path)


def test_duration(synthetic_video):
    d = video_duration_sec(synthetic_video)
    assert 9.5 < d < 10.5


def test_iter_chunks_covers_duration(synthetic_video):
    chunks = list(iter_chunks(synthetic_video, chunk_sec=4.0, min_tail_sec=0.5))
    assert chunks[0][0] == 0.0
    # Final chunk's end must reach near the duration.
    assert chunks[-1][1] >= 8.0


def test_extract_frames_count(synthetic_video):
    frames = extract_frames(synthetic_video, 0.0, 4.0, fps=1.0)
    # 4-second window at 1 fps → roughly 4 frames (allow +/-1 for rounding).
    assert 3 <= len(frames) <= 5
    for f in frames:
        assert f.size[0] > 0 and f.size[1] > 0


def test_extract_frames_empty_range(synthetic_video):
    assert extract_frames(synthetic_video, 5.0, 5.0, fps=1.0) == []


def test_extract_frames_missing_file():
    assert extract_frames("/nonexistent.mp4", 0.0, 5.0, fps=1.0) == []
