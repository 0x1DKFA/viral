# Project: Viral Gameplay Clipper
- **Goal**: Automate the creation of 9:16 viral shorts from raw gameplay.
- **Constraints**: 
    - Input: Raw `.mp4` recordings.
    - Audio: Game audio only (no mic/commentary).
    - Hardware: I have an RTX 5070 (use local CUDA/GPU acceleration).
- **Strategy**: Focus on visual cues (UI changes, kill-feeds, high motion) since there is no voice to trigger clips.
- **Tools Preferred**: Python, FFmpeg, OpenCV, and local VLMs (like Qwen2-VL).
