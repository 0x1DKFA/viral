# Design Spec: Viral Gameplay Clipper (The Director)
**Date:** 2026-05-10
**Status:** Draft
**Goal:** Automate the creation of 9:16 viral shorts from raw gameplay using Qwen3-VL-7B-Instruct.

## 1. Overview
A local pipeline that processes raw `.mp4` gameplay recordings, identifies high-action highlights using a Vision Language Model (VLM), and exports them as vertically-cropped (9:16) clips and a combined montage.

## 2. Architecture
The system follows a sequential pipeline:
1.  **Ingestion:** Manually triggered scan of `raw_recordings/`.
2.  **Segmentation:** FFmpeg slices input into 15-30s chunks with a 5s overlap.
3.  **VLM Analysis:** Qwen3-VL-7B-Instruct (4-bit quantized) analyzes segments for "viral potential" and "action focus coordinates".
4.  **Filtering:** Segments scoring above a threshold are selected for export.
5.  **Smart Cropping:** Dynamic 9:16 crop using OpenCV/FFmpeg based on action center coordinates.
6.  **Export:** Individual clips in `output/clips/` and a consolidated `montage.mp4`.

## 3. Tech Stack
- **Language:** Python 3.13
- **VLM:** Qwen3-VL-7B-Instruct (via `transformers`)
- **GPU Acceleration:** NVIDIA RTX 5070 (CUDA 13.1, Flash Attention 2)
- **Video Processing:** FFmpeg (nvenc), OpenCV
- **Quantization:** `bitsandbytes` (4-bit)

## 4. Folder Structure
```
viral/
├── raw_recordings/   # Input
├── processed/        # Archive
├── segments/         # Temporary cache
├── output/
│   ├── clips/        # 9:16 vertical shorts
│   └── montage.mp4   # Final reel
├── src/
│   ├── main.py       # Orchestrator
│   ├── analyzer.py   # VLM interface
│   └── processor.py  # Video editing logic
├── docs/             # Documentation and specs
└── requirements.txt
```

## 5. Success Criteria
- Successfully load and run Qwen3-VL-7B-Instruct on the RTX 5070.
- Extract clips that contain visually interesting action (kills, high motion, UI events).
- Produce 9:16 vertical video that centers the action correctly.
- Pipeline runs start-to-finish without crashing on 1080p/4k input.

## 6. Implementation Plan (High Level)
1. Install system dependencies (FFmpeg).
2. Set up Python virtual environment and install ML libraries.
3. Implement `processor.py` for segmentation and cropping.
4. Implement `analyzer.py` for Qwen3-VL inference.
5. Implement `main.py` to glue the pipeline together.
6. Verify with `Metro Redux` sample recording.
