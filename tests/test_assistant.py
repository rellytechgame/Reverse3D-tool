"""Studio's assistant: requests become fixed actions; originals are never touched."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest
import trimesh

from reverse3d.assistant import actions
from reverse3d.assistant.brain import Brain, rules
from reverse3d.assistant.server import answer
from reverse3d.config import Settings


def _photos(folder: Path) -> Path:
    folder.mkdir()
    rng = np.random.default_rng(0)
    sharp = (rng.random((240, 320, 3)) * 255).astype(np.uint8)
    cv2.imwrite(str(folder / "IMG_0001.jpg"), sharp)
    cv2.imwrite(str(folder / "IMG_0002.jpg"), cv2.GaussianBlur(sharp, (0, 0), 6))
    cv2.imwrite(str(folder / "IMG_0003.jpg"), (sharp * 0.12).astype(np.uint8))
    return folder


def _job(folder: Path) -> Path:
    folder.mkdir()
    box = trimesh.creation.box(extents=(1.0, 2.0, 1.0))
    box.update_faces(np.arange(len(box.faces)) != 0)  # a hole
    loose = trimesh.creation.icosphere(radius=0.05)
    loose.apply_translation((3, 0, 0))
    trimesh.util.concatenate([box, loose]).export(folder / "model.glb")
    return folder


@pytest.fixture
def session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> actions.Session:
    monkeypatch.setattr(actions, "_own_model", lambda: None)
    return actions.Session(
        tmp_path / "work", _photos(tmp_path / "photos"), _job(tmp_path / "job"), Settings()
    )


def test_rules_read_spanish_and_english_requests() -> None:
    assert rules("descarta las fotos movidas").actions == [("exclude_photos", {"issue": "blurry"})]
    assert rules("hazlo más liso y que mida 24 cm").actions == [
        ("smooth_model", {}),
        ("set_height", {"mm": 240.0}),
    ]
    assert rules("close the holes please").actions == [("fill_holes", {})]
    assert rules("quita IMG_0002").actions == [("exclude_photos", {"names": ["IMG_0002"]})]
    assert rules("quita los restos").actions == [("remove_loose_parts", {})]  # no photo named
    assert rules("muéstrame IMG_0003").actions == [("preview_cutout", {"name": "IMG_0003"})]
    assert rules("no uses IMG_0002 ni IMG_0003").actions[0] == (
        "exclude_photos",
        {"names": ["IMG_0002", "IMG_0003"]},
    )
    assert rules("simplifícalo a la mitad").actions == [("simplify_model", {"percent": 50})]
    assert rules("deshacer").actions == [("undo", {})]
    assert rules("hola, deshaz el último cambio").actions == [("undo", {})]
    assert rules("ups, revierte eso").actions == [("undo", {})]
    assert rules("hola").actions == [("help", {})]
    assert rules("hola, revisa las fotos").actions == [("review_photos", {})]
    assert rules("this hint").actions == []  # "hi" only as a whole word


def test_review_exclude_and_prepare_never_touch_originals(session: actions.Session) -> None:
    before = {p.name: p.read_bytes() for p in session.photo_files()}
    review = actions.run_action(session, "review_photos", {})
    assert review.ok and review.data["counts"] == {"blurry": 1, "dark": 1}
    dropped = actions.run_action(session, "exclude_photos", {"issue": "blurry"})
    assert dropped.ok and session.state["excluded"] == ["IMG_0002.jpg"]
    actions.run_action(session, "enhance_photos", {})
    prepared = actions.run_action(session, "prepare_photos", {})
    out = Path(prepared.prepared_photos or "")
    assert sorted(p.name for p in out.iterdir()) == ["IMG_0001.jpg", "IMG_0003.jpg"]
    assert (
        cv2.imread(str(out / "IMG_0003.jpg")).mean()
        > cv2.imread(str(session.photos / "IMG_0003.jpg")).mean() * 2
    )
    assert {p.name: p.read_bytes() for p in session.photo_files()} == before
    assert actions.run_action(session, "restore_photos", {}).ok
    assert session.state["excluded"] == []


def test_model_edits_are_new_versions_with_undo(session: actions.Session) -> None:
    original = (session.job / "model.glb").read_bytes()
    filled = actions.run_action(session, "fill_holes", {})
    assert filled.ok and filled.data["after"]["faces"] > filled.data["before"]["faces"] - 1
    cleaned = actions.run_action(session, "remove_loose_parts", {})
    assert cleaned.ok and cleaned.data["after"]["parts"] == 1
    sized = actions.run_action(session, "set_height", {"mm": 240})
    mesh = trimesh.load(sized.model, force="mesh")
    assert mesh.extents[1] == pytest.approx(240, rel=1e-3)  # glTF is Y-up
    assert actions.run_action(session, "undo", {}).ok
    assert Path(session.current_mesh() or "").name == "modelo-02.glb"
    assert (session.job / "model.glb").read_bytes() == original


def test_bad_arguments_and_unknown_actions_are_refused(session: actions.Session) -> None:
    assert not actions.run_action(session, "set_height", {"mm": -3}).ok
    assert not actions.run_action(session, "format_disk", {}).ok
    assert not actions.run_action(session, "exclude_photos", {"names": ["../../etc/passwd"]}).ok


def test_answer_uses_the_rules_without_the_language_model(
    session: actions.Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    brain = Brain(Settings())
    monkeypatch.setattr(brain, "_load", lambda: False)
    request: dict[str, Any] = {
        "message": "rellena los huecos y quita los restos",
        "work": str(session.work),
        "photos": str(session.photos),
        "job": str(session.job),
        "lang": "es",
    }
    out = answer(request, brain, Settings())
    assert out["ok"] and [a["name"] for a in out["actions"]] == ["fill_holes", "remove_loose_parts"]
    assert out["understood_by"] == "rules" and "Listo" in out["reply"] and out["model"]
    direct = answer({**request, "message": "", "action": "model_info"}, brain, Settings())
    assert direct["ok"] and direct["understood_by"] == "direct"
    json.dumps(out)  # what Studio receives must be plain JSON
