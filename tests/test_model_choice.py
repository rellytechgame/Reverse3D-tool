"""Choosing which Voxel Enterprise model pack to work with."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from reverse3d import ai_models
from reverse3d.errors import InputValidationError


def _pack(name: str, version: str, tier: str = "free", usable: bool = True) -> dict[str, Any]:
    return {
        "path": f"{name}-{version}.r3dm",
        "name": name,
        "version": version,
        "tier": tier,
        "tasks": ["depth", "mask", "scene"],
        "usable": usable,
    }


PACKS = [
    _pack("optiverse-1", "0.1.0"),
    _pack("optiverse-1", "0.2.0"),
    _pack("optiverse-1", "0.10.0"),
    _pack("optiverse-2-plus", "0.1.0"),
    _pack("revenus-1", "0.2.0", tier="pro", usable=False),
]


def test_auto_prefers_newest_generation_and_version() -> None:
    assert ai_models.choose(PACKS, "auto")["name"] == "optiverse-2-plus"
    only_one = [p for p in PACKS if p["name"] == "optiverse-1"]
    assert ai_models.choose(only_one, "auto")["version"] == "0.10.0"  # not text order


def test_auto_prefers_professional_when_the_licence_allows_it() -> None:
    packs = [*PACKS[:-1], _pack("revenus-1", "0.2.0", tier="pro")]
    assert ai_models.choose(packs, "auto")["name"] == "revenus-1"


def test_a_chosen_pack_or_version_wins_and_falls_back_when_missing() -> None:
    assert ai_models.choose(PACKS, "optiverse-1@0.1.0")["version"] == "0.1.0"
    assert ai_models.choose(PACKS, "optiverse-1")["version"] == "0.10.0"
    # not allowed by the licence (or uninstalled): automatic choice instead
    assert ai_models.choose(PACKS, "revenus-1")["name"] == "optiverse-2-plus"
    assert ai_models.choose([], "auto") is None


def test_use_model_remembers_the_choice(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    class FakeCore:
        @staticmethod
        def list_packs() -> list[dict[str, Any]]:
            return [dict(p) for p in PACKS]

    monkeypatch.setenv("REVERSE3D_HOME", str(tmp_path))
    monkeypatch.delenv(ai_models.MODEL_ENV, raising=False)
    monkeypatch.setattr(ai_models, "_core_models", lambda: FakeCore)
    assert ai_models.selected_model() == "auto"
    result = ai_models.use_model("Optiverse-1@0.1.0")
    assert result["selected"] == "optiverse-1@0.1.0"
    listing = ai_models.list_models()
    active = [m for m in listing["models"] if m["active"]]  # type: ignore[union-attr]
    assert listing["selected"] == "optiverse-1@0.1.0" and active[0]["version"] == "0.1.0"
    monkeypatch.setenv(ai_models.MODEL_ENV, "optiverse-1@0.2.0")  # one run overrides it
    assert ai_models.selected_model() == "optiverse-1@0.2.0"
    with pytest.raises(InputValidationError):
        ai_models.use_model("revenus-1")
