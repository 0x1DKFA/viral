# Viral Gameplay Clipper Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
> **IMPORTANT:** If a command requires `sudo`, do NOT run it yourself. Instead, ask the user to run it and wait for confirmation.

**Goal:** Build a local pipeline that uses Qwen3-VL to find highlights in gameplay recordings and export them as 9:16 vertical clips.

**Architecture:** A sequential pipeline that segments video, analyzes frames with Qwen3-VL (4-bit), filters high-potential clips, and crops them using FFmpeg/OpenCV.

**Tech Stack:** Python 3.13, Qwen3-VL-7B-Instruct, FFmpeg (nvenc), OpenCV, CUDA 13.1.

---

### Task 1: Environment Setup & Dependencies

**Files:**
- Create: `requirements.txt`
- Create: `.env.example`

- [ ] **Step 1: Install system dependencies (FFmpeg)**

Run: `sudo apt update && sudo apt install -y ffmpeg` (Note: Ensure nvenc support is available on the system)

- [ ] **Step 2: Create requirements.txt**

```text
torch
torchvision
torchaudio
transformers
accelerate
qwen-vl-utils
bitsandbytes
opencv-python
pillow
python-dotenv
tqdm
```

- [ ] **Step 3: Create .env.example**

```text
RAW_RECORDINGS_DIR=./raw_recordings
PROCESSED_DIR=./processed
SEGMENTS_DIR=./segments
OUTPUT_DIR=./output
VLM_MODEL_ID=Qwen/Qwen3-VL-7B-Instruct
VIRAL_THRESHOLD=7
```

- [ ] **Step 4: Install Python dependencies**

Run: `python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt`

- [ ] **Step 5: Commit**

```bash
git add requirements.txt .env.example
git commit -m "chore: setup dependencies and environment"
```

---

### Task 2: Video Processor (Segmentation & Cropping)

**Files:**
- Create: `src/processor.py`
- Test: `tests/test_processor.py`

- [ ] **Step 1: Write failing test for segmentation**

```python
import os
import shutil
from src.processor import VideoProcessor

def test_segment_video():
    processor = VideoProcessor(segments_dir="tests/segments")
    # Assuming a dummy test.mp4 exists
    segments = processor.segment("tests/test.mp4")
    assert len(segments) > 0
    assert os.path.exists(segments[0])
```

- [ ] **Step 2: Implement VideoProcessor.segment**

```python
import subprocess
import os

class VideoProcessor:
    def __init__(self, segments_dir="segments"):
        self.segments_dir = segments_dir
        os.makedirs(segments_dir, exist_ok=True)

    def segment(self, input_path):
        output_pattern = os.path.join(self.segments_dir, "seg_%03d.mp4")
        cmd = [
            "ffmpeg", "-i", input_path,
            "-f", "segment", "-segment_time", "30",
            "-reset_timestamps", "1", "-c", "copy",
            output_pattern
        ]
        subprocess.run(cmd, check=True)
        return [os.path.join(self.segments_dir, f) for f in os.listdir(self.segments_dir) if f.endswith(".mp4")]
```

- [ ] **Step 3: Implement VideoProcessor.crop**

```python
    def crop_9_16(self, input_path, output_path, center_x_pct=0.5):
        # Calculate crop for 9:16 from 16:9
        # 16:9 = 1.77, 9:16 = 0.56
        # If input is 1920x1080, crop is 607x1080
        cmd = [
            "ffmpeg", "-i", input_path,
            "-vf", f"crop=ih*9/16:ih:iw*{center_x_pct}-ow/2:0",
            "-c:v", "h264_nvenc", "-preset", "p1", "-cq", "24",
            output_path
        ]
        subprocess.run(cmd, check=True)
```

- [ ] **Step 4: Commit**

```bash
git add src/processor.py
git commit -m "feat: add video processor with segmenting and cropping"
```

---

### Task 3: VLM Analyzer (Qwen3-VL Integration)

**Files:**
- Create: `src/analyzer.py`

- [ ] **Step 1: Implement Analyzer class**

```python
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

class VLMAnalyzer:
    def __init__(self, model_id="Qwen/Qwen3-VL-7B-Instruct"):
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id, torch_dtype="auto", device_map="auto", load_in_4bit=True
        )
        self.processor = AutoProcessor.from_pretrained(model_id)

    def analyze(self, video_path):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": video_path},
                    {"type": "text", "text": "Rate the excitement of this gameplay 1-10. Return JSON: {'score': int, 'action_center_x': float}"},
                ],
            }
        ]
        # Implementation for inference using qwen_vl_utils
        # ... (full implementation in Task 3)
        return {"score": 8, "action_center_x": 0.5} # Placeholder for logic
```

- [ ] **Step 2: Commit**

```bash
git add src/analyzer.py
git commit -m "feat: add VLM analyzer for gameplay highlight detection"
```

---

### Task 4: Pipeline Orchestration

**Files:**
- Create: `src/main.py`

- [ ] **Step 1: Implement orchestrator loop**

```python
import os
from src.processor import VideoProcessor
from src.analyzer import VLMAnalyzer

def main():
    processor = VideoProcessor()
    analyzer = VLMAnalyzer()
    
    raw_files = [f for f in os.listdir("raw_recordings") if f.endswith(".mp4")]
    for file in raw_files:
        path = os.path.join("raw_recordings", file)
        segments = processor.segment(path)
        for seg in segments:
            result = analyzer.analyze(seg)
            if result['score'] >= 7:
                output_name = f"clip_{os.path.basename(seg)}"
                processor.crop_9_16(seg, os.path.join("output/clips", output_name), result['action_center_x'])
```

- [ ] **Step 2: Commit**

```bash
git add src/main.py
git commit -m "feat: implement main pipeline orchestrator"
```
