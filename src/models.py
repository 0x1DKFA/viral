from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field


_ALLOWED_SCOUT_TYPES = frozenset({
    "firefight", "multi_kill", "clutch", "ace", "movement", "boss_kill",
    "glitch", "fail", "jumpscare", "big_damage", "reaction", "other",
})


@dataclass
class ScoutRegion:
    start_sec: float   # absolute time in source (or window-relative before pipeline offsets)
    end_sec: float
    type: str          # one of _ALLOWED_SCOUT_TYPES; unknown values coerced to "other"


@dataclass
class Highlight:
    is_highlight: bool
    score: int
    start_sec: float
    end_sec: float
    action_center_x: float
    title: str
    description: str
    hashtags: list[str] = field(default_factory=list)
    # Only populated when --explain-sizing is on; surfaces the model's own
    # reasoning about clip length, useful for auditing whether sizing is adaptive.
    recommended_duration_sec: float | None = None
    sizing_reason: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


def _coerce_bool(v) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return v != 0
    if isinstance(v, str):
        return v.strip().lower() in {"true", "yes", "1"}
    return False


def _coerce_int(v, default: int = 0) -> int:
    try:
        return int(round(float(v)))
    except (TypeError, ValueError):
        return default


def _coerce_float(v, default: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _coerce_tags(v) -> list[str]:
    if isinstance(v, list):
        return [str(t).strip() for t in v if str(t).strip()]
    if isinstance(v, str):
        parts = re.split(r"[,\s]+", v)
        return [p for p in parts if p]
    return []


def parse_highlight(
    raw: str,
    chunk_duration_sec: float,
    min_clip_sec: float = 0.0,
) -> Highlight | None:
    """Parse VLM output into a Highlight. Returns None on unrecoverable failure.

    If `min_clip_sec > 0` and the model's window is shorter, expand it symmetrically
    (clamped to the chunk). If the chunk itself is shorter than min_clip_sec,
    accept the full chunk rather than rejecting the highlight.
    """
    if not raw or not raw.strip():
        return None

    text = raw.strip()
    # Try direct JSON first, then fall back to regex extraction.
    candidates: list[str] = [text]
    m = _JSON_BLOCK.search(text)
    if m and m.group() != text:
        candidates.append(m.group())

    # Tolerate Python-style single quotes by swapping when double-quote parse fails.
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
        return None

    score = _coerce_int(data.get("score"), default=0)
    score = max(0, min(10, score))

    start = _coerce_float(data.get("start_sec"), default=0.0)
    end = _coerce_float(data.get("end_sec"), default=chunk_duration_sec)
    start = max(0.0, min(chunk_duration_sec, start))
    end = max(0.0, min(chunk_duration_sec, end))
    if end <= start:
        # Try to recover by using the full chunk; if that's still invalid, give up.
        start, end = 0.0, chunk_duration_sec
        if end <= start:
            return None

    if min_clip_sec > 0 and (end - start) < min_clip_sec:
        # Expand the window symmetrically, then push off whichever wall we hit.
        pad = (min_clip_sec - (end - start)) / 2.0
        start = max(0.0, start - pad)
        end = min(chunk_duration_sec, end + pad)
        if (end - start) < min_clip_sec:
            if start <= 0.0:
                end = min(chunk_duration_sec, start + min_clip_sec)
            else:
                start = max(0.0, end - min_clip_sec)
        # If the chunk itself is shorter than min_clip_sec, accept the whole chunk
        # rather than dropping a valid highlight. Rare in practice but worth handling.

    cx = _coerce_float(data.get("action_center_x"), default=0.5)
    cx = max(0.0, min(1.0, cx))

    is_highlight = _coerce_bool(data.get("is_highlight", score > 0))

    title = str(data.get("title", "")).strip()
    description = str(data.get("description", "")).strip()
    hashtags = _coerce_tags(data.get("hashtags"))

    rec_dur: float | None = None
    if "recommended_duration_sec" in data:
        try:
            rec_dur = float(data["recommended_duration_sec"])
        except (TypeError, ValueError):
            rec_dur = None
    sizing_reason = str(data.get("sizing_reason", "")).strip()

    return Highlight(
        is_highlight=is_highlight,
        score=score,
        start_sec=start,
        end_sec=end,
        action_center_x=cx,
        title=title,
        description=description,
        hashtags=hashtags,
        recommended_duration_sec=rec_dur,
        sizing_reason=sizing_reason,
    )


_SLUG_BAD = re.compile(r"[^a-z0-9]+")


def slugify(text: str, max_len: int = 50) -> str:
    s = _SLUG_BAD.sub("-", text.lower()).strip("-")
    return s[:max_len] or "untitled"
