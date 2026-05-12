import json

from src.models import Highlight, parse_highlight, slugify


def test_parse_valid_json():
    raw = json.dumps({
        "is_highlight": True,
        "score": 8,
        "start_sec": 5.0,
        "end_sec": 15.0,
        "action_center_x": 0.6,
        "title": "Triple kill",
        "description": "Quick triple in mid.",
        "hashtags": ["#valorant", "#triple"],
    })
    h = parse_highlight(raw, chunk_duration_sec=60.0)
    assert isinstance(h, Highlight)
    assert h.score == 8 and h.is_highlight
    assert h.start_sec == 5.0 and h.end_sec == 15.0
    assert h.action_center_x == 0.6
    assert h.hashtags == ["#valorant", "#triple"]


def test_parse_single_quotes_with_preamble():
    raw = "Sure, here is the JSON: {'is_highlight': true, 'score': 9, 'start_sec': 0, 'end_sec': 10, 'action_center_x': 0.5, 'title': 'Ace', 'description': 'Clutch.', 'hashtags': ['#ace']}"
    h = parse_highlight(raw, chunk_duration_sec=60.0)
    assert h is not None
    assert h.score == 9 and h.title == "Ace"


def test_clamping():
    raw = json.dumps({
        "is_highlight": True,
        "score": 99,
        "start_sec": -3,
        "end_sec": 999,
        "action_center_x": 1.7,
        "title": "x",
        "description": "y",
        "hashtags": [],
    })
    h = parse_highlight(raw, chunk_duration_sec=60.0)
    assert h is not None
    assert h.score == 10
    assert h.start_sec == 0.0
    assert h.end_sec == 60.0
    assert h.action_center_x == 1.0


def test_invalid_returns_none():
    assert parse_highlight("not json", chunk_duration_sec=60.0) is None
    assert parse_highlight("", chunk_duration_sec=60.0) is None


def test_missing_fields_default_gracefully():
    raw = json.dumps({"score": 7, "start_sec": 1, "end_sec": 4})
    h = parse_highlight(raw, chunk_duration_sec=60.0)
    assert h is not None
    assert h.score == 7
    assert h.action_center_x == 0.5
    assert h.hashtags == []


def test_end_before_start_recovers_or_rejects():
    raw = json.dumps({"score": 5, "start_sec": 20, "end_sec": 10})
    h = parse_highlight(raw, chunk_duration_sec=60.0)
    assert h is not None
    assert h.start_sec == 0.0 and h.end_sec == 60.0


def test_slugify():
    assert slugify("Clutch 1v3 Ace!") == "clutch-1v3-ace"
    assert slugify("") == "untitled"
    assert slugify("a" * 200, max_len=10) == "a" * 10
