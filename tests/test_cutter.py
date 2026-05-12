import os
import shutil
import subprocess

import pytest

from src.cutter import cut_and_crop


@pytest.fixture(scope="module")
def synthetic_1080p(tmp_path_factory):
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not available")
    path = tmp_path_factory.mktemp("vids") / "src.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y", "-f", "lavfi",
            "-i", "testsrc=duration=10:size=1920x1080:rate=30",
            "-pix_fmt", "yuv420p", str(path),
        ],
        check=True, capture_output=True,
    )
    return str(path)


def _probe_size(path: str) -> tuple[int, int]:
    out = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height", "-of", "csv=p=0:s=x", path,
        ],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    w, h = out.split("x")
    return int(w), int(h)


def test_cut_and_crop_produces_9_16(tmp_path, synthetic_1080p):
    if shutil.which("ffprobe") is None:
        pytest.skip("ffprobe not available")
    out = tmp_path / "out.mp4"
    cut_and_crop(synthetic_1080p, str(out), start_sec=1.0, end_sec=3.0, center_x_pct=0.5, use_nvenc=False)
    assert out.exists() and out.stat().st_size > 0
    w, h = _probe_size(str(out))
    # 1080 * 9/16 = 607.5 → ffmpeg rounds to even, so expect ~608.
    assert h == 1080
    assert abs(w - 608) <= 2


def test_invalid_range_raises(tmp_path, synthetic_1080p):
    out = tmp_path / "out.mp4"
    with pytest.raises(ValueError):
        cut_and_crop(synthetic_1080p, str(out), start_sec=5.0, end_sec=5.0, use_nvenc=False)
