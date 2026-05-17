import json

import pytest

from src.models import ScoutRegion
from src.scout import (
    iter_scout_windows,
    merge_regions,
    parse_scout_regions,
    split_long_regions,
)


# ---------------------- parse_scout_regions ----------------------

def test_parse_valid_regions():
    raw = json.dumps({
        "regions": [
            {"start_sec": 5.0, "end_sec": 18.0, "type": "firefight"},
            {"start_sec": 30.0, "end_sec": 45.0, "type": "multi_kill"},
        ]
    })
    regions = parse_scout_regions(raw, window_duration_sec=60.0)
    assert len(regions) == 2
    assert regions[0].type == "firefight"
    assert regions[1].start_sec == 30.0


def test_parse_unknown_type_becomes_other():
    raw = json.dumps({"regions": [{"start_sec": 1, "end_sec": 5, "type": "epic_moment"}]})
    regions = parse_scout_regions(raw, window_duration_sec=60.0)
    assert len(regions) == 1
    assert regions[0].type == "other"


def test_parse_clamps_to_window():
    raw = json.dumps({"regions": [{"start_sec": -5, "end_sec": 200, "type": "ace"}]})
    regions = parse_scout_regions(raw, window_duration_sec=60.0)
    assert len(regions) == 1
    assert regions[0].start_sec == 0.0
    assert regions[0].end_sec == 60.0


def test_parse_rejects_end_le_start():
    raw = json.dumps({"regions": [
        {"start_sec": 10, "end_sec": 10, "type": "fail"},
        {"start_sec": 20, "end_sec": 15, "type": "fail"},
        {"start_sec": 30, "end_sec": 35, "type": "fail"},
    ]})
    regions = parse_scout_regions(raw, window_duration_sec=60.0)
    assert len(regions) == 1
    assert (regions[0].start_sec, regions[0].end_sec) == (30.0, 35.0)


def test_parse_empty_regions_list():
    raw = json.dumps({"regions": []})
    assert parse_scout_regions(raw, window_duration_sec=60.0) == []


def test_parse_missing_regions_key():
    raw = json.dumps({"other": "stuff"})
    assert parse_scout_regions(raw, window_duration_sec=60.0) == []


def test_parse_with_preamble_and_single_quotes():
    raw = (
        "Sure, here is the JSON: "
        "{'regions': [{'start_sec': 1.0, 'end_sec': 4.0, 'type': 'reaction'}]}"
    )
    regions = parse_scout_regions(raw, window_duration_sec=60.0)
    assert len(regions) == 1
    assert regions[0].type == "reaction"


def test_parse_garbage_returns_empty():
    assert parse_scout_regions("definitely not json", window_duration_sec=60.0) == []
    assert parse_scout_regions("", window_duration_sec=60.0) == []


# ---------------------- merge_regions ----------------------

def test_merge_disjoint_unchanged():
    regions = [
        ScoutRegion(0, 10, "firefight"),
        ScoutRegion(50, 60, "ace"),
    ]
    merged = merge_regions(regions, merge_gap_sec=2.0)
    assert len(merged) == 2


def test_merge_overlapping_combines():
    regions = [
        ScoutRegion(0, 15, "firefight"),
        ScoutRegion(10, 25, "firefight"),
    ]
    merged = merge_regions(regions)
    assert len(merged) == 1
    assert (merged[0].start_sec, merged[0].end_sec) == (0, 25)


def test_merge_near_adjacent_combines():
    # Gap is 1.5s, below default 2.0 merge_gap_sec.
    regions = [
        ScoutRegion(0, 10, "firefight"),
        ScoutRegion(11.5, 20, "firefight"),
    ]
    merged = merge_regions(regions, merge_gap_sec=2.0)
    assert len(merged) == 1
    assert (merged[0].start_sec, merged[0].end_sec) == (0, 20)


def test_merge_far_apart_stays_separate():
    regions = [
        ScoutRegion(0, 10, "firefight"),
        ScoutRegion(15, 25, "firefight"),  # 5s gap > 2s
    ]
    merged = merge_regions(regions, merge_gap_sec=2.0)
    assert len(merged) == 2


def test_merge_unsorted_input():
    regions = [
        ScoutRegion(50, 60, "ace"),
        ScoutRegion(0, 10, "firefight"),
        ScoutRegion(5, 15, "firefight"),
    ]
    merged = merge_regions(regions, merge_gap_sec=2.0)
    assert len(merged) == 2
    assert merged[0].start_sec == 0
    assert merged[0].end_sec == 15
    assert merged[1].start_sec == 50


def test_merge_empty():
    assert merge_regions([]) == []


# ---------------------- split_long_regions ----------------------

def test_split_short_unchanged():
    regions = [ScoutRegion(0, 30, "firefight")]
    out = split_long_regions(regions, max_sec=90.0, overlap_sec=10.0)
    assert out == regions


def test_split_long_region():
    regions = [ScoutRegion(0, 200, "firefight")]
    out = split_long_regions(regions, max_sec=90.0, overlap_sec=10.0)
    # 200s region with max_sec=90, step=80: subs at [0,90), [80,170), [160,200).
    assert len(out) == 3
    assert out[0].start_sec == 0 and out[0].end_sec == 90
    assert out[-1].end_sec == 200
    # All sub-regions keep the original type.
    assert all(r.type == "firefight" for r in out)


def test_split_propagates_type():
    regions = [ScoutRegion(0, 300, "boss_kill")]
    out = split_long_regions(regions, max_sec=90.0, overlap_sec=10.0)
    assert all(r.type == "boss_kill" for r in out)


def test_split_invalid_overlap_raises():
    with pytest.raises(ValueError):
        split_long_regions([ScoutRegion(0, 200, "fail")], max_sec=10.0, overlap_sec=10.0)


# ---------------------- iter_scout_windows ----------------------

def test_iter_windows_short_video():
    windows = list(iter_scout_windows(duration_sec=100.0, window_sec=300.0))
    assert windows == [(0.0, 100.0)]


def test_iter_windows_overlap():
    windows = list(iter_scout_windows(duration_sec=900.0, window_sec=300.0, overlap_sec=60.0))
    # step = 240. Windows: [0,300), [240,540), [480,780), final anchored to [600, 900).
    assert windows[0] == (0.0, 300.0)
    assert windows[-1][1] == 900.0
    # Each window must be exactly 300 long, and overlap correctly.
    for w in windows:
        assert w[1] - w[0] == pytest.approx(300.0)


def test_iter_windows_zero_duration():
    assert list(iter_scout_windows(duration_sec=0.0)) == []


def test_iter_windows_invalid_overlap():
    with pytest.raises(ValueError):
        list(iter_scout_windows(duration_sec=500, window_sec=300, overlap_sec=300))
