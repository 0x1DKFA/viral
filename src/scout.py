"""Scout-pass utilities: prompt, region parser, window iterator, merge/split helpers.

The scout pass walks the source video in large overlapping windows and asks the VLM
to identify candidate viral regions in absolute time. A subsequent detail pass scores
and labels each region. This module owns everything *except* the VLM call itself —
that lives on VLMAnalyzer.scout() so model loading is shared with the detail pass.
"""

from __future__ import annotations

import json
import re
from typing import Iterator

from src.models import ScoutRegion, _ALLOWED_SCOUT_TYPES


SCOUT_PROMPT = (
    "You are scouting a {window_duration:.1f}-second window of gameplay video for "
    "viral-worthy moments. Frames are sampled in chronological order at low rate.\n"
    "\n"
    "Identify ALL distinct moments that might be worth posting as short-form clips. "
    "For each, return its start and end seconds (relative to this window's start) "
    "and a tag describing what kind of moment it is.\n"
    "\n"
    "Allowed type tags (use the closest fit):\n"
    "  firefight, multi_kill, clutch, ace, movement, boss_kill,\n"
    "  glitch, fail, jumpscare, big_damage, reaction, other\n"
    "\n"
    "Be inclusive. A detail pass will score and refine each region; over-flagging "
    "is cheap, missing real moments is not. But do NOT flag mundane gameplay "
    "(walking, looting, menus, loading screens, downtime).\n"
    "\n"
    "If a moment is clearly ongoing at the window boundary, extend the region to "
    "the boundary — don't trim it because the window cuts off.\n"
    "\n"
    "Return ONLY a JSON object:\n"
    '{{\n'
    '  "regions": [\n'
    '    {{"start_sec": float, "end_sec": float, "type": string}},\n'
    "    ...\n"
    "  ]\n"
    "}}\n"
    'If no viral moments, return: {{"regions": []}}\n'
    "\n"
    "Output ONLY the JSON object, nothing else."
)

SCOUT_RETRY_SUFFIX = (
    "\n\nYour previous response was not valid JSON. Reply with ONLY the JSON object, "
    "no prose, no markdown fencing."
)


_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


def parse_scout_regions(raw: str, window_duration_sec: float) -> list[ScoutRegion]:
    """Parse a scout response into a list of ScoutRegion. Returns [] on failure.

    Tolerates single-quoted JSON and surrounding prose (same shape as parse_highlight).
    Clamps each region to [0, window_duration_sec]; rejects regions where end <= start.
    Unknown type strings are coerced to "other".
    """
    if not raw or not raw.strip():
        return []

    text = raw.strip()
    candidates: list[str] = [text]
    m = _JSON_BLOCK.search(text)
    if m and m.group() != text:
        candidates.append(m.group())

    data: dict | None = None
    for candidate in candidates:
        for attempt in (candidate, candidate.replace("'", '"')):
            try:
                parsed = json.loads(attempt)
                if isinstance(parsed, dict):
                    data = parsed
                    break
            except json.JSONDecodeError:
                continue
        if data is not None:
            break

    if data is None:
        return []

    raw_regions = data.get("regions")
    if not isinstance(raw_regions, list):
        return []

    out: list[ScoutRegion] = []
    for entry in raw_regions:
        if not isinstance(entry, dict):
            continue
        try:
            start = float(entry.get("start_sec", 0.0))
            end = float(entry.get("end_sec", 0.0))
        except (TypeError, ValueError):
            continue
        start = max(0.0, min(window_duration_sec, start))
        end = max(0.0, min(window_duration_sec, end))
        if end <= start:
            continue
        type_raw = str(entry.get("type", "other")).strip().lower()
        if type_raw not in _ALLOWED_SCOUT_TYPES:
            type_raw = "other"
        out.append(ScoutRegion(start_sec=start, end_sec=end, type=type_raw))

    return out


def iter_scout_windows(
    duration_sec: float, window_sec: float = 300.0, overlap_sec: float = 60.0
) -> Iterator[tuple[float, float]]:
    """Yield (start, end) windows covering [0, duration_sec] with overlap.

    The last window is anchored to the duration so the tail is always covered.
    Step is `window_sec - overlap_sec`.
    """
    if duration_sec <= 0:
        return
    if window_sec <= 0:
        raise ValueError("window_sec must be > 0")
    if overlap_sec < 0 or overlap_sec >= window_sec:
        raise ValueError("overlap_sec must be in [0, window_sec)")

    step = window_sec - overlap_sec
    if duration_sec <= window_sec:
        yield (0.0, duration_sec)
        return

    starts: list[float] = []
    t = 0.0
    while t + window_sec < duration_sec:
        starts.append(t)
        t += step
    # Final window anchored to the tail.
    starts.append(max(0.0, duration_sec - window_sec))

    seen: set[tuple[float, float]] = set()
    for start in starts:
        end = min(duration_sec, start + window_sec)
        key = (round(start, 3), round(end, 3))
        if key in seen:
            continue
        seen.add(key)
        yield (start, end)


def merge_regions(
    regions: list[ScoutRegion], merge_gap_sec: float = 2.0
) -> list[ScoutRegion]:
    """Merge overlapping or near-adjacent regions. Stable on type via first-wins."""
    if not regions:
        return []
    sorted_r = sorted(regions, key=lambda r: r.start_sec)
    merged: list[ScoutRegion] = [sorted_r[0]]
    for r in sorted_r[1:]:
        last = merged[-1]
        if r.start_sec - last.end_sec < merge_gap_sec:
            merged[-1] = ScoutRegion(
                start_sec=last.start_sec,
                end_sec=max(last.end_sec, r.end_sec),
                type=last.type,
            )
        else:
            merged.append(r)
    return merged


def split_long_regions(
    regions: list[ScoutRegion], max_sec: float = 90.0, overlap_sec: float = 10.0
) -> list[ScoutRegion]:
    """Split any region longer than `max_sec` into overlapping sub-regions.

    Bounds the detail-pass cost per region. Dedup at cut time handles near-duplicate
    Highlights produced by the overlapping sub-regions.
    """
    if max_sec <= 0:
        return list(regions)
    if overlap_sec < 0 or overlap_sec >= max_sec:
        raise ValueError("overlap_sec must be in [0, max_sec)")

    step = max_sec - overlap_sec
    out: list[ScoutRegion] = []
    for r in regions:
        length = r.end_sec - r.start_sec
        if length <= max_sec:
            out.append(r)
            continue
        t = r.start_sec
        while t < r.end_sec:
            sub_end = min(r.end_sec, t + max_sec)
            out.append(ScoutRegion(start_sec=t, end_sec=sub_end, type=r.type))
            if sub_end >= r.end_sec:
                break
            t += step
    return out
