"""Fixture pack for labeled Colonist screenshots.

Each fixture is a paired <name>.png screenshot and <name>.json label file
stored under tests/fixtures/colonist/.  The label JSON records the ground
truth that the CV pipeline should recover from the screenshot.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from ..board.board import Resource
from ..board.hex_grid import AXIAL_POSITIONS
from .schema import PlayerColor


FIXTURE_DIR = Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "colonist"


def _hex_key(hex_coord: tuple[int, int]) -> str:
    return f"{hex_coord[0]},{hex_coord[1]}"


@dataclass
class PieceLabel:
    """Ground-truth pieces for one player color."""

    color: str
    settlements: list[list[float]] = field(default_factory=list)
    cities: list[list[float]] = field(default_factory=list)
    # Each road is [[x1,y1], [x2,y2]] in canonical vertex coordinates.
    roads: list[list[list[float]]] = field(default_factory=list)


@dataclass
class FixtureLabel:
    """Ground truth for a single Colonist screenshot."""

    name: str
    tags: list[str] = field(default_factory=list)

    # Board layout (null when not labeled)
    resources_by_hex: Optional[dict[str, str]] = None
    numbers_by_hex: Optional[dict[str, Optional[int]]] = None
    desert_hex: Optional[list[int]] = None

    # Public pieces
    pieces: list[PieceLabel] = field(default_factory=list)
    robber_hex: Optional[list[int]] = None

    # Turn context
    phase: Optional[str] = None
    current_player_color: Optional[str] = None
    prompt_text: Optional[str] = None

    # Private hand (only if screenshot shows our cards)
    hand_resources: Optional[dict[str, int]] = None

    # Capture metadata
    resolution: Optional[list[int]] = None
    notes: str = ""


def save_label(label: FixtureLabel, path: Path) -> None:
    path.write_text(json.dumps(asdict(label), indent=2) + "\n")


def load_label(path: Path) -> FixtureLabel:
    raw = json.loads(path.read_text())
    pieces = [PieceLabel(**p) for p in raw.pop("pieces", [])]
    return FixtureLabel(**raw, pieces=pieces)


def iter_fixtures(fixture_dir: Path | None = None) -> list[tuple[Path, FixtureLabel]]:
    """Return all (screenshot_path, label) pairs found in the fixture directory."""
    root = fixture_dir or FIXTURE_DIR
    results: list[tuple[Path, FixtureLabel]] = []
    for label_path in sorted(root.glob("*.json")):
        image_path = label_path.with_suffix(".png")
        if not image_path.exists():
            continue
        results.append((image_path, load_label(label_path)))
    return results


def blank_label(name: str, *, tags: list[str] | None = None) -> FixtureLabel:
    """Create a label template with empty fields ready for manual annotation."""
    resources = {_hex_key(h): "" for h in AXIAL_POSITIONS}
    numbers: dict[str, Optional[int]] = {_hex_key(h): None for h in AXIAL_POSITIONS}
    hand = {r.value: 0 for r in Resource if r != Resource.DESERT}
    return FixtureLabel(
        name=name,
        tags=tags or [],
        resources_by_hex=resources,
        numbers_by_hex=numbers,
        desert_hex=None,
        pieces=[PieceLabel(color=c.value) for c in PlayerColor],
        robber_hex=None,
        phase=None,
        current_player_color=None,
        hand_resources=hand,
    )
