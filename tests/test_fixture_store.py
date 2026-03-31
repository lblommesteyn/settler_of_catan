"""Tests for the fixture store: schema round-trip, loader, and synthetic fixture validation."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from catan.board.hex_grid import AXIAL_POSITIONS
from catan.colonist_cv.fixture_store import (
    FIXTURE_DIR,
    _hex_key,
    blank_label,
    iter_fixtures,
    load_label,
    save_label,
)


def test_label_round_trip(tmp_path: Path):
    label = blank_label("test_round_trip", tags=["setup", "1080p"])
    label.phase = "setup"
    label.robber_hex = [-2, 1]
    label.resolution = [1920, 1080]
    label.pieces[0].settlements = [[0.866, 0.5]]

    path = tmp_path / "test_round_trip.json"
    save_label(label, path)
    loaded = load_label(path)

    assert loaded.name == "test_round_trip"
    assert loaded.tags == ["setup", "1080p"]
    assert loaded.phase == "setup"
    assert loaded.robber_hex == [-2, 1]
    assert loaded.resolution == [1920, 1080]
    assert loaded.pieces[0].settlements == [[0.866, 0.5]]
    assert loaded.pieces[0].color == "red"
    assert len(loaded.pieces) == 5
    assert loaded.resources_by_hex is not None
    assert len(loaded.resources_by_hex) == len(AXIAL_POSITIONS)


def test_blank_label_has_all_hex_keys():
    label = blank_label("check_keys")
    assert label.resources_by_hex is not None
    assert label.numbers_by_hex is not None
    for h in AXIAL_POSITIONS:
        key = _hex_key(h)
        assert key in label.resources_by_hex
        assert key in label.numbers_by_hex


def test_iter_fixtures_finds_paired_files(tmp_path: Path):
    # Create a valid pair
    label = blank_label("alpha")
    save_label(label, tmp_path / "alpha.json")
    cv2.imwrite(str(tmp_path / "alpha.png"), np.zeros((100, 100, 3), dtype=np.uint8))

    # Create an orphan JSON with no matching PNG — should be skipped
    save_label(blank_label("orphan"), tmp_path / "orphan.json")

    fixtures = iter_fixtures(tmp_path)
    assert len(fixtures) == 1
    assert fixtures[0][0].name == "alpha.png"
    assert fixtures[0][1].name == "alpha"


def test_iter_fixtures_returns_empty_for_missing_dir(tmp_path: Path):
    assert iter_fixtures(tmp_path / "nonexistent") == []


# -- Synthetic fixture validation (only runs if the fixture has been generated) --

SYNTHETIC_LABEL = FIXTURE_DIR / "synthetic_setup.json"
SYNTHETIC_IMAGE = FIXTURE_DIR / "synthetic_setup.png"


@pytest.mark.skipif(
    not SYNTHETIC_IMAGE.exists(),
    reason="Run 'python tests/generate_synthetic_fixture.py' to create the synthetic fixture",
)
class TestSyntheticFixture:

    def test_label_loads(self):
        label = load_label(SYNTHETIC_LABEL)
        assert label.name == "synthetic_setup"
        assert "synthetic" in label.tags
        assert label.desert_hex == [-2, 1]

    def test_image_loads_and_matches_resolution(self):
        label = load_label(SYNTHETIC_LABEL)
        bgr = cv2.imread(str(SYNTHETIC_IMAGE))
        assert bgr is not None
        h, w = bgr.shape[:2]
        assert label.resolution == [w, h]

    def test_resource_labels_are_complete(self):
        label = load_label(SYNTHETIC_LABEL)
        assert label.resources_by_hex is not None
        assert len(label.resources_by_hex) == len(AXIAL_POSITIONS)
        assert all(v != "" for v in label.resources_by_hex.values())

    def test_number_labels_are_complete(self):
        label = load_label(SYNTHETIC_LABEL)
        assert label.numbers_by_hex is not None
        desert_key = _hex_key((-2, 1))
        for key, val in label.numbers_by_hex.items():
            if key == desert_key:
                assert val is None
            else:
                assert isinstance(val, int) and 2 <= val <= 12

    def test_fixture_appears_in_iter(self):
        fixtures = iter_fixtures()
        names = [label.name for _, label in fixtures]
        assert "synthetic_setup" in names
