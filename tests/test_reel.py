"""Pure-python tests for reel helpers — no ffmpeg required."""
import json
import os
from unittest.mock import patch

import pytest

from src.reel import (
    MIN_CLIPS_FOR_REEL,
    _load_sidecars,
    build_reel,
)


def _make_sidecar(
    dir_path: str,
    stem: str,
    abs_start: float,
    abs_end: float,
    score: int = 8,
    region_type: str = "firefight",
    make_mp4: bool = True,
):
    sidecar = os.path.join(dir_path, stem + ".json")
    mp4 = os.path.join(dir_path, stem + ".mp4")
    with open(sidecar, "w", encoding="utf-8") as f:
        json.dump(
            {
                "source_path": "/fake.mp4",
                "region_start_sec": abs_start - 1.0,
                "region_end_sec": abs_end + 1.0,
                "region_type": region_type,
                "absolute_start_sec": abs_start,
                "absolute_end_sec": abs_end,
                "highlight": {
                    "score": score, "title": "x", "description": "y",
                    "is_highlight": True, "start_sec": 0, "end_sec": abs_end - abs_start,
                    "action_center_x": 0.5, "hashtags": [],
                },
            },
            f,
        )
    if make_mp4:
        with open(mp4, "wb") as f:
            f.write(b"fake mp4 content")
    return sidecar, mp4


def test_load_sidecars_sorts_by_collection_order(tmp_path):
    # _load_sidecars itself returns in glob-sorted (filename) order; build_reel
    # sorts by absolute_start_sec. This test just confirms collection works.
    _make_sidecar(tmp_path, "002__score-08__b", abs_start=20, abs_end=30)
    _make_sidecar(tmp_path, "001__score-07__a", abs_start=5, abs_end=15)
    entries = _load_sidecars(str(tmp_path))
    assert len(entries) == 2


def test_load_sidecars_skips_missing_mp4(tmp_path):
    _make_sidecar(tmp_path, "001__score-08__a", abs_start=5, abs_end=15)
    _make_sidecar(tmp_path, "002__score-09__b", abs_start=20, abs_end=30, make_mp4=False)
    entries = _load_sidecars(str(tmp_path))
    assert len(entries) == 1
    assert entries[0].absolute_start_sec == 5.0


def test_load_sidecars_skips_malformed(tmp_path):
    _make_sidecar(tmp_path, "001__score-08__a", abs_start=5, abs_end=15)
    bad = tmp_path / "002__bad.json"
    bad.write_text("{this is not json")
    (tmp_path / "002__bad.mp4").write_bytes(b"x")
    entries = _load_sidecars(str(tmp_path))
    assert len(entries) == 1


def test_load_sidecars_skips_missing_keys(tmp_path):
    # sidecar missing absolute_start_sec/absolute_end_sec must be skipped.
    bad = tmp_path / "001__no_keys.json"
    bad.write_text(json.dumps({"highlight": {"score": 8}}))
    (tmp_path / "001__no_keys.mp4").write_bytes(b"x")
    entries = _load_sidecars(str(tmp_path))
    assert entries == []


def test_build_reel_skips_when_too_few_clips(tmp_path):
    # 1 clip = no reel
    _make_sidecar(tmp_path, "001__score-08__a", abs_start=5, abs_end=15)
    result = build_reel(str(tmp_path), source_path="/fake.mp4", landscape=False)
    assert result is None
    assert not (tmp_path / "_reel.mp4").exists()


def test_build_reel_chronological_order(tmp_path):
    # 3 clips in non-chronological filename order; build_reel must sort by abs_start.
    _make_sidecar(tmp_path, "002__score-08__b", abs_start=100, abs_end=110)
    _make_sidecar(tmp_path, "003__score-09__c", abs_start=50, abs_end=60)
    _make_sidecar(tmp_path, "001__score-07__a", abs_start=200, abs_end=210)

    captured_paths: list[list[str]] = []

    def fake_concat(clip_paths, out_path):
        captured_paths.append(list(clip_paths))
        # Pretend the reel was built.
        with open(out_path, "wb") as f:
            f.write(b"fake reel")

    with patch("src.reel._concat_lossless", side_effect=fake_concat):
        result = build_reel(str(tmp_path), source_path="/fake.mp4", landscape=False)

    assert result is not None
    assert result.endswith("_reel.mp4")
    assert len(captured_paths) == 1
    paths = captured_paths[0]
    # Order must reflect absolute_start_sec ascending: 50, 100, 200.
    assert "003__score-09__c.mp4" in paths[0]  # abs_start=50
    assert "002__score-08__b.mp4" in paths[1]  # abs_start=100
    assert "001__score-07__a.mp4" in paths[2]  # abs_start=200


def test_build_reel_routes_landscape(tmp_path):
    _make_sidecar(tmp_path, "001__score-08__a", abs_start=5, abs_end=15)
    _make_sidecar(tmp_path, "002__score-09__b", abs_start=20, abs_end=30)

    called = {"vertical": 0, "landscape": 0}

    def fake_vertical(clip_paths, out_path):
        called["vertical"] += 1

    def fake_landscape(source_path, entries, out_path):
        called["landscape"] += 1

    with patch("src.reel._concat_lossless", side_effect=fake_vertical), \
         patch("src.reel._build_landscape", side_effect=fake_landscape):
        build_reel(str(tmp_path), source_path="/fake.mp4", landscape=False)
        build_reel(str(tmp_path), source_path="/fake.mp4", landscape=True)

    assert called == {"vertical": 1, "landscape": 1}


def test_min_clips_for_reel_is_two():
    # Guard against accidental change of the min-clip threshold.
    assert MIN_CLIPS_FOR_REEL == 2
