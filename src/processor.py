import subprocess
import os

class VideoProcessor:
    def __init__(self, segments_dir="segments"):
        self.segments_dir = segments_dir
        os.makedirs(segments_dir, exist_ok=True)

    def segment(self, input_path):
        output_pattern = os.path.join(self.segments_dir, "seg_%03d.mp4")
        # -f segment splits the video
        # -segment_time 30 is the target length
        # -reset_timestamps 1 ensures segments start at 0
        # -c copy is fast (no re-encoding)
        cmd = [
            "ffmpeg", "-y", "-i", input_path,
            "-f", "segment", "-segment_time", "30",
            "-reset_timestamps", "1", "-c", "copy",
            output_pattern
        ]
        subprocess.run(cmd, check=True)
        return sorted([os.path.join(self.segments_dir, f) for f in os.listdir(self.segments_dir) if f.endswith(".mp4")])

    def crop_9_16(self, input_path, output_path, center_x_pct=0.5):
        # Calculate crop for 9:16 from 16:9
        # Filter: crop=w:h:x:y
        # w = ih * 9/16
        # h = ih
        # x = (iw * center_x_pct) - (w / 2)
        # y = 0
        crop_filter = f"crop=ih*9/16:ih:iw*{center_x_pct}-ow/2:0"
        
        cmd = [
            "ffmpeg", "-y", "-i", input_path,
            "-vf", crop_filter,
            "-c:v", "h264_nvenc", "-preset", "p1", "-cq", "24",
            "-c:a", "copy",
            output_path
        ]
        subprocess.run(cmd, check=True)
