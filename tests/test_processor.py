import os
import shutil
import pytest
from src.processor import VideoProcessor

@pytest.fixture
def processor():
    test_segments_dir = "tests/segments"
    if os.path.exists(test_segments_dir):
        shutil.rmtree(test_segments_dir)
    return VideoProcessor(segments_dir=test_segments_dir)

def test_segment_video(processor):
    # This test needs a dummy video file.
    # We can create a 1-second black video using ffmpeg for testing.
    dummy_video = "tests/test_input.mp4"
    if not os.path.exists(dummy_video):
        import subprocess
        subprocess.run([
            "ffmpeg", "-f", "lavfi", "-i", "color=c=black:s=1280x720:d=1",
            "-vcodec", "libx264", "-pix_fmt", "yuv420p", dummy_video
        ], check=True)
    
    segments = processor.segment(dummy_video)
    assert len(segments) > 0
    assert os.path.exists(segments[0])
    
    # Cleanup
    if os.path.exists(dummy_video):
        os.remove(dummy_video)

def test_crop_video(processor):
    dummy_video = "tests/test_crop_input.mp4"
    output_video = "tests/test_crop_output.mp4"
    if not os.path.exists(dummy_video):
        import subprocess
        subprocess.run([
            "ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=blue:s=1920x1080:d=1",
            "-vcodec", "libx264", "-pix_fmt", "yuv420p", dummy_video
        ], check=True)
    
    processor.crop_9_16(dummy_video, output_video, center_x_pct=0.5)
    assert os.path.exists(output_video)
    
    # Cleanup
    if os.path.exists(dummy_video):
        os.remove(dummy_video)
    if os.path.exists(output_video):
        os.remove(output_video)
