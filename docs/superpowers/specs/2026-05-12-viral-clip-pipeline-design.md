# Viral Clip Pipeline — Rework Design

Date: 2026-05-12
Status: Approved (approach A)

## Goal

Trigger a Python script that runs Qwen3-VL over gameplay recordings (mostly under 10 minutes, occasionally up to ~20 minutes) and produces ready-to-post 9:16 short clips with title, description, and hashtags.

Hardware target: RTX 5070 workstation.

## Approach: Single-pass per chunk

For each source video:
1. Walk the video in fixed-length chunks (~60s) without writing intermediate segment files.
2. For each chunk, extract frames in-memory via OpenCV at 1 fps.
3. Send the frames to Qwen3-VL in a single call that returns a complete JSON payload: `is_highlight`, `score`, `start_sec`, `end_sec` (relative to the chunk), `action_center_x`, `title`, `description`, `hashtags`.
4. If `is_highlight` and `score >= threshold`, run one ffmpeg call that cuts AND crops the highlight directly from the original source.
5. Write the cropped clip plus a JSON sidecar containing the model's metadata.
6. Archive the original source.

Rejected alternatives:
- **Two-pass scoring → metadata**: doubles GPU work on the chunks that matter most.
- **Whole-video single VLM call**: risks OOM on the 5070 (12GB) for 20-minute videos at usable frame counts.

## Module layout

```
src/
  __init__.py
  __main__.py      # `python -m src`
  main.py          # thin shim → cli.main (preserves muscle memory)
  cli.py           # argparse, dispatches folder vs file mode
  pipeline.py      # orchestrates per-file processing
  sampler.py       # cv2-based in-memory frame extraction + chunk iterator
  analyzer.py      # Qwen3-VL wrapper, returns Highlight
  cutter.py        # ffmpeg cut + crop in a single call
  models.py        # Highlight dataclass + parser
```

Each module has one clear job:
- `sampler.extract_frames(path, start_sec, end_sec, fps)` — list of PIL frames.
- `sampler.iter_chunks(path, chunk_sec)` — `(start, end)` ranges covering the whole video.
- `analyzer.VLMAnalyzer.analyze(frames, chunk_duration_sec) -> Highlight | None`.
- `cutter.cut_and_crop(src, out, start_sec, end_sec, center_x_pct)` — one ffmpeg call.
- `pipeline.process_file(src, out_dir, analyzer, cfg)` — orchestration.
- `cli.main()` — parses args, builds config, picks mode.

## VLM contract

One call per chunk. Frames are passed via `process_vision_info` with defensive unpacking for both 2-tuple and 3-tuple returns (this is the fix for the current `ValueError: not enough values to unpack`).

Returned JSON schema:
```json
{
  "is_highlight": true,
  "score": 8,
  "start_sec": 12.5,
  "end_sec": 22.0,
  "action_center_x": 0.55,
  "title": "Clutch 1v3 ace with pistol",
  "description": "Pulled off a 1v3 with only a pistol after losing the round economy.",
  "hashtags": ["#valorant", "#clutch", "#1v3"]
}
```

Validation:
- Extract JSON via regex (`\{.*\}`, DOTALL) to tolerate model preamble/postamble.
- Parse into `Highlight` dataclass; coerce types where safe.
- Clamp `start_sec`/`end_sec` to `[0, chunk_duration]`; reject if `end <= start`.
- Clamp `action_center_x` to `[0, 1]`; default to `0.5` if missing.
- On parse failure, retry once with a stricter "ONLY a JSON object, no prose" suffix. Still failing → log and skip the chunk.

## CLI

```
python -m src                        # folder mode using RAW_RECORDINGS_DIR
python -m src path/to/video.mp4      # single-file mode
python -m src path/to/dir/           # custom folder
python src/main.py [...]             # equivalent thin shim
```

Flags:
- `--out PATH` — output directory (default `OUTPUT_DIR` env or `./output`).
- `--threshold INT` — minimum score to keep (default 7).
- `--chunk SEC` — chunk length in seconds (default 60).
- `--fps FLOAT` — frame sampling rate within a chunk (default 1.0).
- `--model ID` — Hugging Face model id (default `VLM_MODEL_ID` env or `Qwen/Qwen3-VL-8B-Instruct`).
- `--keep-source` — do not move processed sources to `PROCESSED_DIR`.
- `--dry-run` — score chunks and print results, do not cut clips.

## Output layout

```
output/
  <source-stem>/
    001__score-09__clutch-1v3-ace.mp4
    001__score-09__clutch-1v3-ace.json
    002__score-07__triple-kill.mp4
    002__score-07__triple-kill.json
```

Filename pattern: `{chunk_idx:03d}__score-{score:02d}__{title-slug}.mp4`. Sidecar JSON contains the full `Highlight` plus `source_path`, `chunk_start_sec`, `chunk_end_sec`.

Cheap resume: skip a chunk if its `.mp4` output already exists.

## Error handling

- Source unreadable → log, skip file, continue.
- Chunk frame extraction returns empty list → log, skip chunk.
- VLM parse failure after one retry → log, skip chunk.
- ffmpeg cut/crop nonzero exit → log stderr, skip chunk (do not abort the file).
- Model load failure (e.g., OOM, bad CUDA) → fatal, abort run.

## Testing

- `tests/test_models.py` — `Highlight` parsing: valid JSON, JSON with single quotes, JSON with surrounding prose, malformed, missing fields, type coercion, clamping.
- `tests/test_sampler.py` — chunk iterator covers full duration with no gaps; frame extraction returns the expected count for a synthetic ffmpeg-generated video.
- `tests/test_cutter.py` — cut + crop produces an output file at the expected duration and 9:16 aspect from a synthetic source.
- `tests/test_analyzer.py` — kept skipped behind a GPU marker; covers model load only.

## Dependencies

`requirements.txt` stays mostly the same. Drop nothing (qwen-vl-utils still used, but with defensive tuple unpacking). No new top-level deps.

## Migration

Old files removed and replaced:
- `src/main.py` → rewritten as thin shim.
- `src/processor.py` → removed (replaced by `sampler.py` + `cutter.py`).
- `src/analyzer.py` → rewritten.

No segment files are ever written to disk. The `SEGMENTS_DIR` env var becomes a no-op (kept in `.env.example` for back-compat but unused).
