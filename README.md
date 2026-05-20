# viral

A scout + detail VLM pipeline that turns long gameplay recordings into ready-to-post 9:16 short clips and a stitched highlight reel.

Built around Qwen3-VL running locally on a single GPU. Given an `.mp4` (or a directory of them), it:

1. **Scout pass** — walks the video in 5-min overlapping windows and asks the VLM to flag every region that might be viral (firefights, multi-kills, clutches, fails, etc.) with rough timestamps and a type tag.
2. **Merge + split** — joins regions that span window boundaries; caps very long regions for the next stage.
3. **Detail pass** — re-samples each region at higher fps/resolution and asks the VLM to score it, refine the boundaries, and return title / description / hashtags / horizontal action center.
4. **Cut + crop** — produces a 9:16 vertical clip per kept region with `h264_nvenc` (or `libx264` fallback) plus a JSON sidecar containing the full metadata.
5. **Reel** — concatenates the clips into a single `_reel.mp4` per source video, 9:16 by default or 16:9 with `--reel-landscape`.

The scout/detail split exists so an engagement that straddles a chunk boundary still gets recognised as one region and cut as one clip with proper lead-in and aftermath.

---

## Requirements

**Hardware**
- NVIDIA GPU with ~12 GB VRAM (RTX 4070/5070/3080 class works). Falls back to CPU encode if `h264_nvenc` is unavailable.
- ~16 GB system RAM is workable thanks to `low_cpu_mem_usage=True` model loading.

**Software**
- Python 3.10+
- `ffmpeg` and `ffprobe` on `PATH`
- CUDA-capable PyTorch (matches your CUDA install)

Python packages (`requirements.txt`):

```
torch, torchvision, torchaudio   # CUDA build
transformers
accelerate
qwen-vl-utils
bitsandbytes                     # 4-bit quantization
opencv-python
pillow
python-dotenv
tqdm
```

---

## Setup

```bash
git clone <repo>
cd viral

python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

cp .env.example .env             # optional: edit defaults
```

Drop `.mp4` files into `./raw_recordings/` (or pass paths directly).

---

## Usage

```bash
# Process every .mp4 in $RAW_RECORDINGS_DIR
python -m src

# One file at a time
python -m src path/to/video.mp4

# A different directory
python -m src path/to/recordings/

# Dry run — scout, score, log decisions; do not cut clips
python -m src --dry-run

# Verbose — DEBUG-level logging plus library noise
python -m src --debug
```

Equivalent entry points: `python -m src`, `python src/main.py`. Same flags.

---

## Output layout

```
output/
└── <source-stem>/
    ├── 001__score-09__clutch-1v3-ace.mp4   ← 9:16 vertical clip
    ├── 001__score-09__clutch-1v3-ace.json  ← sidecar metadata
    ├── 002__score-07__triple-kill.mp4
    ├── 002__score-07__triple-kill.json
    └── _reel.mp4                            ← stitched highlight reel
```

Sidecar shape:

```json
{
  "source_path": "/abs/path/to/source.mp4",
  "region_start_sec": 142.5,
  "region_end_sec": 178.0,
  "region_type": "firefight",
  "absolute_start_sec": 145.0,
  "absolute_end_sec": 172.0,
  "highlight": {
    "is_highlight": true,
    "score": 9,
    "start_sec": 2.5,
    "end_sec": 29.5,
    "action_center_x": 0.55,
    "title": "Clutch 1v3 ace",
    "description": "Pulled off a 1v3 with only a pistol after losing the round economy.",
    "hashtags": ["#valorant", "#clutch", "#1v3"]
  }
}
```

Re-running the pipeline on the same source is safe: clips whose chunk index already has a `.mp4` are skipped, and prior `absolute_start_sec`/`absolute_end_sec` ranges are loaded into the dedup table.

Processed source files are moved to `./processed/` after a successful run (override with `--keep-source`).

---

## CLI options — full reference

Run `python -m src --help` for the parser's own copy. The grouped reference below adds context on when to reach for each.

### Input / output

| Flag | Default | Notes |
|---|---|---|
| `path` (positional) | `$RAW_RECORDINGS_DIR` or `./raw_recordings` | File or directory. Single-file mode bypasses the folder convention. |
| `--out PATH` | `$OUTPUT_DIR` or `./output` | Per-source subdirectory is created inside this. |
| `--threshold INT` | `$VIRAL_THRESHOLD` or `7` | Minimum detail-pass score (1-10) to save a clip. Raise to be pickier. |
| `--model ID` | `$VLM_MODEL_ID` or `Qwen/Qwen3-VL-8B-Instruct` | Any HF VLM with the same Qwen-VL interface. `Qwen/Qwen3-VL-4B-Instruct` runs in less VRAM. |
| `--keep-source` | off | Don't move processed `.mp4` files to `./processed/`. |
| `--dry-run` | off | Run scout + detail and log every decision; skip clip writes and the reel. Useful for tuning thresholds without burning disk. |

### Scout pass (whole-video boundary finder)

The scout is run at low fps/low resolution so it can see the whole video without OOMing. It outputs candidate regions in absolute source time.

| Flag | Default | Notes |
|---|---|---|
| `--scout-window SEC` | `300` (5 min) | Length of each scout window. Larger = fewer windows but more frames per call. |
| `--scout-overlap SEC` | `60` | How much consecutive scout windows overlap. The overlap is what lets a fight straddling a window boundary get stitched back together by `merge_regions`. |
| `--scout-fps FLOAT` | `1.0` | Frames per second sampled during scout. Higher catches sub-second moments at the cost of VRAM and runtime. |
| `--scout-frame-pixels INT` | `360*640 ≈ 230400` | Per-frame pixel budget for scout. Each scout frame is downscaled until `w*h <= this`. Lower for less VRAM, higher if the scout is missing HUD detail. |
| `--no-scout` | off | Bypass scout entirely. Whole source becomes one region and `split_long_regions` chops it into `--max-region` sub-regions. Use when scout under-flags (e.g. sustained boss-fight content). |

### Detail pass (per-region scoring + metadata)

Run on each region the scout flagged (after merge + split). Higher fps and resolution than scout.

| Flag | Default | Notes |
|---|---|---|
| `--detail-fps FLOAT` | `2.0` | Frames per second sampled within each region. Drop to `1.5` if 90-second regions OOM. |
| `--detail-frame-pixels INT` | `480*854 ≈ 410k` | Per-frame pixel budget for detail (~480p). Lower if VRAM is tight; raise if the model is missing HUD detail (kill feed text, health bars). |

### Region shaping

Operations between scout and detail.

| Flag | Default | Notes |
|---|---|---|
| `--max-region SEC` | `90` | Any merged region longer than this is split into overlapping sub-regions before detail. Bounds detail-pass cost per region. |
| `--region-pad SEC` | `2` | Padding added to each side of a scout region before extracting detail frames. Gives the detail pass a margin to refine inside. |

### Clip shaping

What the detail pass is allowed to return.

| Flag | Default | Notes |
|---|---|---|
| `--min-clip SEC` | `8` | Minimum saved clip length. The detail pass returns its natural window; if it's shorter than this, the parser pads symmetrically (clamped to the region) — the model is not asked to "stretch to N seconds". |
| `--max-clip SEC` | `25` | Cap on detail-pass window length. Raise to allow long-form clips (sustained firefights). |
| `--explain-sizing` | off | Ask the VLM to also return `recommended_duration_sec` and `sizing_reason` in each sidecar. Use to audit whether clip length is actually adaptive across regions or stuck at the floor. |

### Highlight reel

Stitched together from sidecars at the end of each source. Lives at `output/<source-stem>/_reel.mp4`.

| Flag | Default | Notes |
|---|---|---|
| `--reel-landscape` | off (9:16) | Build the reel at source aspect (16:9 for typical gameplay capture). The clips are re-cut from the original source, then concatenated. Without the flag, the existing 9:16 cropped clips are concatenated losslessly. |
| `--no-reel` | off | Skip the reel build. Clips are still produced. |

The reel is only built when there are ≥2 clips for the source.

### Logging

| Flag | Default | Notes |
|---|---|---|
| `--log-dir PATH` | `$LOG_DIR` or `./logs` | A per-run timestamped log file is created here. Both stdout and the file get the same lines. |
| `--log-level LEVEL` | `INFO` | One of DEBUG, INFO, WARNING, ERROR. |
| `--debug` | off | Shorthand for `--log-level DEBUG` plus unmuting noisy libraries (transformers, accelerate, PIL, …). |

---

## Environment variables (`.env`)

These act as defaults; CLI flags override.

| Variable | Default | Used by |
|---|---|---|
| `RAW_RECORDINGS_DIR` | `./raw_recordings` | Default input directory when `path` is omitted. |
| `PROCESSED_DIR` | `./processed` | Where source files are archived after a successful run. |
| `OUTPUT_DIR` | `./output` | Where per-source clip directories are written. |
| `VLM_MODEL_ID` | `Qwen/Qwen3-VL-8B-Instruct` | HF model id. |
| `VIRAL_THRESHOLD` | `7` | Minimum detail-pass score. |
| `LOG_DIR` | `./logs` | Log file destination. |
| `LOG_LEVEL` | `INFO` | Default log verbosity. |

---

## Common tuning scenarios

| Symptom | Lever |
|---|---|
| OOM at model load | Use a smaller `--model`, e.g. `Qwen/Qwen3-VL-4B-Instruct`. |
| OOM during detail pass | `--detail-fps 1.5` or `--detail-frame-pixels 230400` (~360p). |
| OOM during scout | `--scout-fps 0.3` or `--scout-window 240`. |
| Too many false positives | Raise `--threshold 8` (or `9`). |
| Clips are too short / cut too tight | Raise `--max-clip` (the model can pick longer windows). |
| Clips cluster at exactly `--min-clip` | Turn on `--explain-sizing` to see if the model is actually adapting or just defaulting. |
| Engagements still get chopped | Increase `--scout-overlap` so fights at window boundaries get more redundant coverage. |
| Scout returns zero regions on a video that obviously has highlights | Try `--no-scout` to bypass scout and let the detail pass run on every ~90s sub-region. Also check the WARNING log line that shows raw scout output. |
| Want a YouTube-ready recap | `--reel-landscape`. |
| Same input, want to compare two settings | `--keep-source --dry-run` first; the source stays in place so you can rerun. |

---

## Architecture at a glance

```
            ┌──────────────────────────────────────────┐
            │ scout (low fps, low res, 5-min windows)  │
            │   ↓ regions in window-relative time      │
            │   ↓ offset to absolute source time       │
            └──────────────────┬───────────────────────┘
                               ▼
            merge_regions  ──▶ split_long_regions
                               ▼
       ┌──────────────────────────────────────────────────┐
       │ for each region:                                  │
       │   pad ±region_pad → extract detail frames @ fps   │
       │   analyzer.analyze(...) → Highlight               │
       │   threshold + dedup → cut_and_crop → sidecar      │
       └──────────────────────────────────────────────────┘
                               ▼
                         build_reel (9:16 or 16:9)
                               ▼
                         archive source
```

Module layout:

```
src/
├── __main__.py     # python -m src entrypoint
├── main.py         # python src/main.py shim
├── cli.py          # argparse + setup_logging
├── pipeline.py     # process_file: orchestrates scout → detail → cut → reel
├── scout.py        # SCOUT_PROMPT, parse_scout_regions, iter_scout_windows,
│                   #   merge_regions, split_long_regions
├── analyzer.py     # VLMAnalyzer.scout() and .analyze()
├── sampler.py      # video_duration_sec, extract_frames (cv2-backed)
├── cutter.py       # cut_and_crop (single ffmpeg call: -ss/-to + crop + nvenc)
├── reel.py         # build_reel (9:16 lossless concat or 16:9 re-cut)
├── models.py       # Highlight + ScoutRegion dataclasses + parsers
└── log.py          # setup_logging
```

---

## Tests

Pure-python tests (no GPU, no ffmpeg) run instantly:

```bash
python -m pytest tests/test_models.py tests/test_scout.py tests/test_reel.py -v
```

Sampler/cutter tests need `ffmpeg`, `ffprobe`, and `cv2` on the box:

```bash
python -m pytest tests/test_sampler.py tests/test_cutter.py -v
```

---

## Troubleshooting

- **`ValueError: not enough values to unpack` from `process_vision_info`** — older `qwen-vl-utils` versions return a 2-tuple, newer return a 3-tuple. `analyzer.py` handles both; if you still hit this, `pip install -U qwen-vl-utils`.
- **`KeyError: 'min_clip'`** — pre-`min_clip` analyzer wired against new prompt. `git pull`; this was a transient bug.
- **`Asked to sample 'fps' frames per second but no video metadata was provided`** — the analyzer now passes `video_metadata` explicitly. If the warning persists, your `transformers` version may want a different shape; raise an issue.
- **Reel build skipped** — fewer than 2 clips were saved for that source. Either nothing met the threshold or `--dry-run` was on.
- **`h264_nvenc` not found** — the cutter falls back to `libx264` automatically. Reel landscape mode also auto-detects.

---

## Design notes

- The scout/detail split is the answer to the chunking problem: with fixed 60s chunks, a fight whose buildup is in chunk N and peak in chunk N+1 gets cut twice (or missed) because the model never sees the engagement as one unit. Scout sees the whole video in 5-min windows with overlap, and `merge_regions` stitches cross-boundary detections.
- Frames are extracted in-memory via OpenCV. No segment `.mp4` files are written to disk anywhere in the pipeline.
- Dedup is IoU + gap based (`pipeline._is_duplicate`). With the scout-merge step it's mostly redundant but kept as a safety net.
- The detail prompt explicitly says "we'll pad to `min_clip` on our side, you do NOT need to stretch your window" — so `--min-clip 20` doesn't make the model refuse genuine 8-second highlights.
