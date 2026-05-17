# Scout + Detail Architecture

Date: 2026-05-17
Status: Approved (user confirmed chunking-boundary failure case)

## Context

The fixed-60s-chunk loop has a structural failure mode: when a viral engagement straddles a chunk boundary, neither chunk sees the full engagement. The model in each chunk picks a sub-window of the visible portion. Concretely: a firefight whose buildup is in chunk 2 and whose peak is in chunk 3 produces a clip from chunk 3 only, missing the buildup. Confirmed by the user against the actual source.

Patching this with overlap reduces but doesn't eliminate the problem and still asks the model to "pick the highlight within this arbitrary 60s box," which forces narrow selection on sustained engagements (the 8s-of-30s symptom).

The fix: let the VLM decide where the boundaries are. Two-stage scout → detail.

## Architecture

```
source video
  ↓
iter_scout_windows  (5-min windows, 60s overlap)
  ↓
for each scout window:
  extract_frames @ scout_fps (0.5) and scout_frame_pixels (~240p)
  analyzer.scout(frames, window_offset) → list[ScoutRegion]    # absolute time
  ↓
flatten + merge_regions  (merge overlapping or near-adjacent across windows)
  ↓
split_long_regions  (cap detail-pass cost; very long regions become sub-regions)
  ↓
for each region:
  pad ±2s, clamped to [0, source_duration]
  extract_frames @ detail_fps (2.0) and max_frame_pixels (~480p)
  analyzer.analyze(frames, duration, region_type=region.type) → Highlight
  if is_highlight and score >= threshold:
    cut_and_crop from (region_start + highlight.start_sec, region_start + highlight.end_sec)
    write sidecar
```

Dedup, IoU-based, still runs at the cut step — but mostly redundant now since `merge_regions` deduplicates at the scout layer.

## Schemas

### ScoutRegion (new, in `models.py`)

```python
@dataclass
class ScoutRegion:
    start_sec: float   # absolute time in source
    end_sec: float
    type: str          # one of an allowed vocabulary (see prompt)
```

### Scout JSON output

```json
{
  "regions": [
    {"start_sec": 12.5, "end_sec": 38.2, "type": "firefight"},
    {"start_sec": 145.0, "end_sec": 152.7, "type": "multi_kill"}
  ]
}
```

`start_sec` / `end_sec` are window-relative; pipeline adds the window offset to make them absolute. `regions: []` is valid (nothing flagged).

### Highlight (unchanged)

Detail pass returns the existing `Highlight` schema. No backward-incompatible change.

## Module changes

### New: `src/scout.py`

- `SCOUT_PROMPT` — see "Prompts" section.
- `parse_scout_regions(raw, window_duration_sec) -> list[ScoutRegion]` — JSON parser, tolerant to single quotes/preamble (same shape as `parse_highlight`). Clamps each region to `[0, window_duration_sec]`. Rejects regions where `end <= start`. Coerces unknown `type` strings to `"other"`.
- `iter_scout_windows(path, window_sec=300, overlap_sec=60) -> Iterator[tuple[float, float]]` — yields `(start, end)` covering the whole video with overlap. Lives here, not `sampler.py`, because it's scout-specific.
- `merge_regions(regions, merge_gap_sec=2.0) -> list[ScoutRegion]` — sort by start, merge if overlapping or within `merge_gap_sec`. When merging, type defaults to the earlier region's type (good enough for v1).
- `split_long_regions(regions, max_sec=90, overlap_sec=10) -> list[ScoutRegion]` — any region > `max_sec` gets split into overlapping sub-regions. Bounds the detail-pass cost per region.

### `src/models.py`

- Add `ScoutRegion` dataclass.
- No changes to `Highlight` or `parse_highlight`.

### `src/analyzer.py`

- Remove `sampling_fps` from `__init__`. It's now per-call.
- `_generate(messages, frame_count, duration_sec, fps_used)` — `fps_used` comes from caller.
- `_build_messages` becomes private builder used by both `scout()` and `analyze()`. Refactor to accept the prompt string and the `fps_used` rather than building the prompt inline.
- Add `scout(frames, duration_sec, fps_used) -> list[ScoutRegion]`. Same retry-once-on-parse-failure shape as `analyze()`. Returns `[]` on parse failure (rather than `None`) so the pipeline can keep going.
- Modify `analyze()` to accept an optional `region_type: str | None = None`. When provided, prepend a one-line hint to the prompt: `"A scout pass tagged this region as: {region_type}. Confirm and refine."`. No other prompt changes.
- Drop `EXPLAIN_SIZING_FRAGMENT`'s "between min and max" clause's reliance on `self.sampling_fps` — irrelevant.

### `src/sampler.py`

- Keep `extract_frames` and `video_duration_sec`. Both still used.
- Remove `iter_chunks` (no longer the primary path; `iter_scout_windows` replaces it). Old tests for `iter_chunks` get rewritten or deleted.

### `src/pipeline.py`

Major rewrite. New shape:

```python
@dataclass
class PipelineConfig:
    out_dir: str
    processed_dir: Optional[str]
    threshold: int = 7
    scout_window_sec: float = 300.0
    scout_overlap_sec: float = 60.0
    scout_fps: float = 0.5
    scout_frame_pixels: int = 240 * 432
    detail_fps: float = 2.0
    detail_frame_pixels: int = 480 * 854
    max_region_sec: float = 90.0
    region_pad_sec: float = 2.0
    keep_source: bool = False
    dry_run: bool = False
```

`process_file` flow:
1. Read source duration.
2. For each scout window: extract frames at `scout_fps`/`scout_frame_pixels`, call `analyzer.scout(...)`, offset returned regions to absolute time, accumulate.
3. `merge_regions(all_regions)`.
4. `split_long_regions(merged, max_sec=cfg.max_region_sec)`.
5. Seed `saved_ranges` from existing sidecars (existing logic).
6. For each region: pad ±`region_pad_sec`, clamp, extract detail frames, `analyzer.analyze(..., region_type=region.type)`, threshold-check, dedup-check, cut, sidecar.
7. Archive source if `not keep_source and not dry_run`.

The existing `_is_duplicate`, `_seed_saved_ranges`, `_output_names`, `_write_sidecar` are kept and reused unchanged.

### `src/cli.py`

Flag changes:
- **Remove** `--chunk` and `--fps` (the latter renamed for clarity).
- **Add** `--scout-window` (300), `--scout-overlap` (60), `--scout-fps` (0.5), `--scout-frame-pixels` (240*432).
- **Add** `--detail-fps` (2.0). Replaces `--fps`.
- **Rename** `--max-frame-pixels` → `--detail-frame-pixels` for parallel naming.
- **Add** `--max-region` (90) — cap on a single region's length before splitting.
- **Add** `--region-pad` (2.0).
- **Keep** `--min-clip`, `--max-clip`, `--threshold`, `--model`, `--keep-source`, `--dry-run`, `--explain-sizing`.

No legacy/escape-hatch flag — if scout misbehaves, we iterate on scout. Falling back to broken chunking is not progress.

## Prompts

### Scout prompt

```
You are scouting a {window_duration:.1f}-second window of gameplay video for
viral-worthy moments. Frames are sampled in chronological order at low rate.

Identify ALL distinct moments that might be worth posting as short-form clips.
For each, return its start and end seconds (relative to this window's start)
and a tag describing what kind of moment it is.

Allowed type tags (use the closest fit):
  firefight, multi_kill, clutch, ace, movement, boss_kill,
  glitch, fail, jumpscare, big_damage, reaction, other

Be inclusive. A detail pass will score and refine each region; over-flagging
is cheap, missing real moments is not. But do NOT flag mundane gameplay
(walking, looting, menus, loading screens, downtime).

If a moment is clearly ongoing at the window boundary, extend the region to
the boundary — don't trim it because the window cuts off.

Return ONLY a JSON object:
{
  "regions": [
    {"start_sec": float, "end_sec": float, "type": string},
    ...
  ]
}
If no viral moments, return: {"regions": []}

Output ONLY the JSON object, nothing else.
```

The "extend to boundary" instruction is what lets the merge step stitch a fight across two scout windows: each window flags its visible portion all the way to the edge, and `merge_regions` joins them.

### Detail prompt

Existing `PROMPT` in `analyzer.py`. Modification: if `region_type` is provided, prepend one line:

```
A coarse scout pass tagged this region as: {region_type}.
Confirm whether it's actually viral, refine the start/end if needed.
```

This narrows the model's focus without locking it in.

## Memory budget

On 12GB VRAM with the 8B in 4-bit (~5-6GB free for activations):

- Scout call: 5min × 0.5 fps × 240p (~100K px/frame) = 150 frames × 100K = 15M px through encoder. Comfortable.
- Detail call: typical region 10-30s × 2 fps × 480p (~410K px/frame) = 20-60 frames × 410K = 8-25M px. Comfortable.
- Worst case for detail (a 90s region post-split): 180 frames × 410K = 74M px. Tighter — verify, drop `--detail-fps` to 1.5 if it OOMs.

Total VLM calls for a typical 15-min video:
- Scout: ceil((900-60)/(300-60)) = 4 windows → 4 calls
- Detail: ~5-15 regions → 5-15 calls
- Total: 9-19 calls (vs. ~15 chunks today). Net similar, with much better selection.

## Edge cases

- **Scout returns empty for all windows** → no clips. Log: `[scout] 0 regions across N windows`. Correct behavior — there were no highlights.
- **Scout returns malformed JSON** → retry once with strict suffix; on second failure, return `[]` for that window. Other windows still process.
- **Region near video boundary** → `pad ±region_pad_sec` clamps to `[0, source_duration]`.
- **Region longer than `max_region_sec`** → `split_long_regions` produces overlapping sub-regions. Dedup at cut time suppresses near-duplicates from sub-regions.
- **Detail says `is_highlight=false`** → scout was too permissive. Skip and log. Not an error.
- **Detail returns a window smaller than `min_clip`** → existing padding in `parse_highlight` expands it.
- **Scout returns a region of `type` we don't recognize** → parser coerces to `"other"`. Detail-pass hint becomes `"...tagged this region as: other"`. Harmless.

## Verification

1. **Pure-Python tests** (no GPU):
   - `parse_scout_regions` — valid JSON, single quotes, missing regions key, malformed entries, unknown type → `"other"`, end ≤ start rejection, clamping.
   - `merge_regions` — disjoint stays disjoint, overlapping merges, near-adjacent (gap < merge_gap_sec) merges, sorted output.
   - `split_long_regions` — short regions untouched, long regions split with overlap, type propagated.
   - `iter_scout_windows` — covers whole video, overlap is correct, last window reaches duration.
2. **End-to-end** on the same video that previously chopped the fight across chunks 2 and 3:
   - Scout should flag the firefight as a region spanning the previous chunk-2 + chunk-3 area.
   - One detail call on that region should produce a single clip that includes the full fight (buildup + peak + reaction).
   - Sidecar's `chunk_start_sec`/`chunk_end_sec` are replaced by `region_start_sec`/`region_end_sec`. Update field names in sidecar accordingly.
3. **Dry-run inspection**:
   ```
   python -m src --dry-run path/to/video.mp4
   ```
   Should log: scout window count, total regions flagged, regions after merge, regions after split, then detail decisions per region.

## Sidecar JSON schema change

```json
{
  "source_path": "...",
  "region_start_sec": 142.5,
  "region_end_sec": 178.0,
  "region_type": "firefight",
  "absolute_start_sec": 145.0,
  "absolute_end_sec": 172.0,
  "highlight": { ... existing Highlight fields ... }
}
```

`chunk_start_sec`/`chunk_end_sec` are removed (chunks no longer exist). Renamed fields surface the scout's contribution.

## Out of scope

- Whole-video single-pass scout (could be a future optimization if scout-chunked turns out reliable).
- Audio cues (silence detection, loudness peaks) as cheap pre-scout filter.
- Cross-region merging at the *detail* stage (rare in practice once scout merge is solid).
- Score-based region prioritization (process top N) — not needed at <20-min video lengths.
