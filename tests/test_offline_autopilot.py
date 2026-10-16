"""Tests for offline mode, the autopilot and the watched inbox."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest
import trimesh
from typer.testing import CliRunner

from reverse3d import autopilot, offline
from reverse3d.agent import execute_tool
from reverse3d.agent.prompts import render_prompt
from reverse3d.cli import app
from reverse3d.config import Settings
from reverse3d.errors import (
    CaptureQualityError,
    DependencyMissingError,
    InputValidationError,
    ProcessingError,
)
from reverse3d.recipes import RecipeOptions, run_recipe
from reverse3d.watch import DONE_DIR, REVIEW_DIR, WatchFolders, _Tracker, scan, watch

runner = CliRunner()
SMALL = "depth-anything/Depth-Anything-V2-Small-hf"
NO_RENDER = RecipeOptions(render=False)


def _fake_snapshot(home: Path, model: str = SMALL) -> Path:
    revision = offline.MODEL_REVISIONS[model]
    snap = home / "models" / ("models--" + model.replace("/", "--")) / "snapshots" / revision
    snap.mkdir(parents=True)
    (snap / "config.json").write_text("{}")
    (snap / "model.safetensors").write_bytes(b"\0" * 2048)
    return snap


def _photo(path: Path, seed: int = 1) -> Path:
    rng = np.random.default_rng(seed)
    img = cv2.GaussianBlur((rng.random((240, 320, 3)) * 255).astype(np.uint8), (5, 5), 0)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)
    return path


# ─── offline mode ─────────────────────────────────────────────────────────────


class TestConnectivity:
    def test_modes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        assert offline.is_offline(Settings(offline="on")) is True
        assert offline.is_offline(Settings(offline="off")) is False
        assert offline.is_offline(Settings(offline="auto")) is False
        monkeypatch.setattr(offline, "_probe", lambda: False)
        assert offline.is_offline(Settings(offline="auto")) is False  # cached "online"
        assert offline.has_internet(refresh=True) is False
        assert offline.is_offline(Settings(offline="auto")) is True

    def test_real_probe_logic(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def refuse(*_a: Any, **_k: Any) -> None:
            raise OSError("network unreachable")

        monkeypatch.setattr(offline.socket, "create_connection", refuse)
        assert offline._reachable(offline._PROBE_HOSTS) is False

        class _Conn:
            def __enter__(self) -> _Conn:
                return self

            def __exit__(self, *_exc: object) -> None:
                return None

        monkeypatch.setattr(offline.socket, "create_connection", lambda *_a, **_k: _Conn())
        assert offline._reachable(offline._PROBE_HOSTS) is True

    def test_offline_setting_is_validated(self) -> None:
        assert Settings.from_env({"REVERSE3D_OFFLINE": "ON"}).offline == "on"
        with pytest.raises(InputValidationError):
            Settings.from_env({"REVERSE3D_OFFLINE": "sometimes"})


class TestModels:
    def test_local_model_is_used_without_network(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        snap = _fake_snapshot(tmp_path / "_r3d_home")
        monkeypatch.setattr(offline, "_download", lambda _m: pytest.fail("must not download"))
        assert offline.local_model_path(SMALL) == snap
        assert offline.resolve_model(SMALL, Settings(offline="on")) == snap

    def test_missing_model_offline_gives_clear_hint(self) -> None:
        with pytest.raises(DependencyMissingError) as err:
            offline.resolve_model(SMALL, Settings(offline="on"))
        assert "reverse3d offline prepare" in (err.value.hint or "")

    def test_missing_model_online_downloads(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[str] = []

        def fake_download(model: str) -> Path:
            calls.append(model)
            return _fake_snapshot(tmp_path / "_r3d_home", model)

        monkeypatch.setattr(offline, "_download", fake_download)
        path = offline.resolve_model(SMALL, Settings(offline="off"))
        assert calls == [SMALL] and path.name == offline.MODEL_REVISIONS[SMALL]

    def test_download_errors_are_reported(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def broken(_m: str) -> Path:
            raise ConnectionError("reset by peer")

        monkeypatch.setattr(offline, "_download", broken)
        with pytest.raises(DependencyMissingError, match="ConnectionError"):
            offline.resolve_model(SMALL, Settings(offline="off"))

    def test_only_allowlisted_models(self) -> None:
        with pytest.raises(InputValidationError):
            offline.resolve_model("evil/model", Settings(offline="off"))

    def test_every_allowlisted_model_is_pinned(self, tmp_path: Path) -> None:
        from reverse3d.reconstruction.depth_estimation import ALLOWED_MODELS

        assert set(ALLOWED_MODELS) == set(offline.MODEL_REVISIONS)
        assert all(len(sha) == 40 for sha in offline.MODEL_REVISIONS.values())
        other = tmp_path / "_r3d_home" / "models" / ("models--" + SMALL.replace("/", "--"))
        (other / "snapshots" / "unreviewed").mkdir(parents=True)
        (other / "snapshots" / "unreviewed" / "model.safetensors").write_bytes(b"x")
        assert offline.local_model_path(SMALL) is None  # only the pinned revision counts

    def test_prepare_without_depth_extra(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(offline, "_depth_stack_installed", lambda: False)
        report = offline.prepare(Settings())
        assert report["ready"] is True
        assert report["models"][0]["status"] == "skipped"

    def test_prepare_downloads_and_is_idempotent(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(offline, "_depth_stack_installed", lambda: True)
        monkeypatch.setattr(
            offline, "_download", lambda m: _fake_snapshot(tmp_path / "_r3d_home", m)
        )
        first = offline.prepare(Settings(offline="off"))
        second = offline.prepare(Settings(offline="off"))
        assert first["models"][0]["status"] == "downloaded" and first["ready"]
        assert second["models"][0]["status"] == "ready"

    def test_prepare_offline_reports_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(offline, "_depth_stack_installed", lambda: True)
        report = offline.prepare(Settings(offline="on"))
        assert report["ready"] is False and report["models"][0]["status"] == "missing"

    def test_status_and_doctor(self, settings: Settings) -> None:
        status = offline.offline_status(settings)
        assert status["components"]["measure_repair_cad"] == {"available": True, "offline": True}
        assert set(status) >= {"internet", "working_offline", "ready_for_offline", "message"}
        from reverse3d.doctor import environment_report

        assert "ready_for_offline" in environment_report(settings)["offline"]  # type: ignore[operator]


# ─── autopilot: detection ─────────────────────────────────────────────────────


class TestDetection:
    def test_plan_photo_and_transparent(self, plan_image: Path, tmp_path: Path) -> None:
        assert autopilot.detect_input(plan_image).kind == "plan"
        assert autopilot.detect_input(_photo(tmp_path / "foto.jpg")).kind == "photo"
        rgba = np.zeros((100, 100, 4), np.uint8)
        rgba[30:70, 30:70] = (40, 120, 200, 255)
        cv2.imwrite(str(tmp_path / "logo.png"), rgba)
        detection = autopilot.detect_input(tmp_path / "logo.png")
        assert detection.kind == "plan" and "transparent" in detection.reason

    def test_folders(self, tmp_path: Path, plan_image: Path) -> None:
        for i in range(3):
            _photo(tmp_path / "taza" / f"img_{i}.jpg", seed=i)
        assert autopilot.detect_input(tmp_path / "taza").kind == "photos"
        _photo(tmp_path / "pocas" / "a.jpg")
        _photo(tmp_path / "pocas" / "b.jpg", seed=2)
        few = autopilot.detect_input(tmp_path / "pocas")
        assert few.kind == "unknown" and "at least 3" in few.reason
        (tmp_path / "uno").mkdir()
        (tmp_path / "uno" / "plano.png").write_bytes(plan_image.read_bytes())
        assert autopilot.detect_input(tmp_path / "uno").kind == "plan"

    def test_meshes_and_clouds(self, box_mesh_path: Path, tmp_path: Path) -> None:
        assert autopilot.detect_input(box_mesh_path).kind == "mesh"
        cloud = tmp_path / "nube.ply"
        trimesh.PointCloud(np.random.default_rng(0).random((50, 3))).export(cloud)
        assert autopilot.detect_input(cloud).kind == "point_cloud"
        mesh_ply = tmp_path / "pieza.ply"
        trimesh.creation.box().export(mesh_ply)
        assert autopilot.detect_input(mesh_ply).kind == "mesh"

    def test_unknown(self, tmp_path: Path) -> None:
        (tmp_path / "notas.txt").write_text("hola")
        assert autopilot.detect_input(tmp_path / "notas.txt").kind == "unknown"
        assert autopilot.detect_input(tmp_path / "no-existe").kind == "unknown"


# ─── autopilot: runs ──────────────────────────────────────────────────────────


class TestAutoRun:
    def test_plan_end_to_end(self, plan_image: Path, tmp_path: Path, settings: Settings) -> None:
        summary = autopilot.auto_run(plan_image, tmp_path / "out", settings, NO_RENDER)
        assert summary["recipe"] == "plan_to_part"
        assert summary["auto"]["detected"]["kind"] == "plan"
        assert summary["auto"]["attempts"] == [{"attempt": "mask=otsu", "ok": True}]
        text = Path(summary["outputs"]["summary"]).read_text(encoding="utf-8")
        assert "Plano, careta o silueta" in text and "--units-per-pixel" in text

    def test_mesh_end_to_end(self, box_mesh_path: Path, tmp_path: Path, settings: Settings) -> None:
        summary = autopilot.auto_run(box_mesh_path, tmp_path / "cad", settings, NO_RENDER)
        assert summary["recipe"] == "scan_to_cad"
        assert summary["dimensions"]["watertight"] is True
        assert Path(summary["outputs"]["cad_openscad"]).is_file()
        assert any("OpenSCAD" in s for s in summary["next_steps"])

    def test_unknown_input(self, tmp_path: Path, settings: Settings) -> None:
        (tmp_path / "x.txt").write_text("?")
        with pytest.raises(InputValidationError):
            autopilot.auto_run(tmp_path / "x.txt", tmp_path / "o", settings)

    def test_retries_with_another_mask_method(
        self,
        plan_image: Path,
        tmp_path: Path,
        settings: Settings,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        real = autopilot.run_recipe
        methods: list[str] = []

        def flaky(recipe: str, src: Path, out: Path, s: Settings, opt: RecipeOptions) -> Any:
            methods.append(opt.mask_method)
            if opt.mask_method == "otsu":
                raise ProcessingError("No contours found.")
            return real(recipe, src, out, s, opt)

        monkeypatch.setattr(autopilot, "run_recipe", flaky)
        summary = autopilot.auto_run(plan_image, tmp_path / "o", settings, NO_RENDER)
        assert methods == ["otsu", "adaptive"]
        assert [a["ok"] for a in summary["auto"]["attempts"]] == [False, True]

    def test_capture_problems_are_not_retried(
        self, tmp_path: Path, settings: Settings, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        for i in range(3):
            _photo(tmp_path / "fotos" / f"{i}.jpg", seed=i)
        calls: list[str] = []

        def blurry(*_a: Any, **_k: Any) -> Any:
            calls.append("x")
            raise CaptureQualityError("Photos are blurry.", hint="Retake them.")

        monkeypatch.setattr(autopilot, "run_recipe", blurry)
        with pytest.raises(CaptureQualityError):
            autopilot.auto_run(tmp_path / "fotos", tmp_path / "o", settings)
        assert calls == ["x"]

    def test_auto_recipe_and_sandboxed_tool(
        self, plan_image: Path, tmp_path: Path, settings: Settings
    ) -> None:
        summary = run_recipe("auto", plan_image, tmp_path / "r", settings, NO_RENDER)
        assert summary["recipe"] == "plan_to_part"
        ok = execute_tool(
            "reverse3d_run_recipe",
            {"recipe": "auto", "input_path": "plan.png", "output_dir": "res", "render": False},
            settings,
        )
        assert ok["ok"] is True
        bad = execute_tool(
            "reverse3d_run_recipe",
            {"recipe": "auto", "input_path": "../plan.png", "output_dir": "res"},
            settings,
        )
        assert bad["error"]["code"] == "security_violation"

    def test_procesar_prompt(self) -> None:
        text = render_prompt("procesar", {"entrada": "bandeja/pieza.stl"})
        assert "recipe=auto" in text and "output_dir='out/pieza'" in text


# ─── watched inbox ────────────────────────────────────────────────────────────


@pytest.fixture
def folders(tmp_path: Path) -> WatchFolders:
    f = WatchFolders(tmp_path / "bandeja", tmp_path / "resultados")
    f.ensure()
    return f


class TestWatch:
    def test_once_processes_and_moves(
        self, folders: WatchFolders, plan_image: Path, settings: Settings
    ) -> None:
        (folders.inbox / "soporte.png").write_bytes(plan_image.read_bytes())
        (folders.inbox / "nota.txt").write_text("no es 3D")
        (folders.inbox / "descarga.png.part").write_bytes(b"...")
        events = watch(folders, settings, once=True)

        done = next(e for e in events if e["event"] == "done")
        assert done["item"] == "soporte.png" and done["kind"] == "plan"
        assert (folders.outbox / "soporte" / "RESUMEN.md").is_file()
        assert (folders.inbox / DONE_DIR / "soporte.png").is_file()

        failed = next(e for e in events if e["event"] == "failed")
        assert failed["item"] == "nota.txt"
        note = folders.inbox / REVIEW_DIR / "nota.txt.LEEME.txt"
        assert "Qué hacer" in note.read_text(encoding="utf-8")
        assert (folders.inbox / "descarga.png.part").exists()  # partial downloads are ignored

    def test_waits_until_copy_finishes(
        self, folders: WatchFolders, box_mesh_path: Path, settings: Settings
    ) -> None:
        (folders.inbox / "caja.stl").write_bytes(box_mesh_path.read_bytes())
        tracker = _Tracker()
        seen: list[str] = []
        assert scan(folders, settings, tracker, on_event=lambda e: seen.append(e["event"])) == []
        events = scan(folders, settings, tracker, on_event=lambda e: seen.append(e["event"]))
        assert events[0]["event"] == "done" and events[0]["recipe"] == "scan_to_cad"
        assert seen == ["detected", "processing", "done"]

    def test_same_name_twice_gets_unique_folders(
        self, folders: WatchFolders, plan_image: Path, settings: Settings
    ) -> None:
        for _ in range(2):
            (folders.inbox / "plano.png").write_bytes(plan_image.read_bytes())
            watch(folders, settings, once=True)
        assert (folders.outbox / "plano").is_dir() and (folders.outbox / "plano-2").is_dir()
        assert (folders.inbox / DONE_DIR / "plano-2.png").is_file()


# ─── CLI ──────────────────────────────────────────────────────────────────────


def test_cli_offline_auto_and_watch(tmp_path: Path, plan_image: Path) -> None:
    status = runner.invoke(app, ["--json", "offline", "status"])
    assert status.exit_code == 0, status.output
    assert json.loads(status.stdout)["result"]["components"]["plan_to_part"]["offline"] is True

    auto = runner.invoke(
        app, ["--json", "-w", str(tmp_path), "auto", str(plan_image), "--no-render"]
    )
    assert auto.exit_code == 0, auto.output
    result = json.loads(auto.stdout)["result"]
    assert Path(result["outputs"]["summary"]).parent == tmp_path / "resultados" / "plan"

    inbox = tmp_path / "bandeja"
    inbox.mkdir(exist_ok=True)
    (inbox / "otro.png").write_bytes(plan_image.read_bytes())
    watched = runner.invoke(app, ["--json", "-w", str(tmp_path), "watch", "--once"])
    assert watched.exit_code == 0, watched.output
    lines = [json.loads(line) for line in watched.stdout.splitlines() if line.strip()]
    assert any(e["event"] == "done" and e["item"] == "otro.png" for e in lines)
