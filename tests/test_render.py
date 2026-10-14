from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from reverse3d.config import Settings
from reverse3d.errors import DependencyMissingError, InputValidationError
from reverse3d.rendering import blender_render as br


class FakeBlender:
    def __init__(self) -> None:
        self.args: list[str] = []
        self.job: dict[str, Any] = {}

    def __call__(
        self, executable: Path, args: list[str], **_: Any
    ) -> subprocess.CompletedProcess[str]:
        self.args = [str(a) for a in args]
        self.job = json.loads(Path(self.args[-1]).read_text(encoding="utf-8"))
        out = Path(self.job["output"])
        frames = self.job["turntable_frames"]
        targets = (
            [out]
            if frames == 0
            else [out.with_name(f"{out.stem}_{i:04d}{out.suffix}") for i in range(frames)]
        )
        for t in targets:
            t.write_bytes(b"\x89PNG")
        stdout = br._RESULT_MARKER + json.dumps(
            {"files": [str(t) for t in targets], "blender": "4.2.0"}
        )
        return subprocess.CompletedProcess(args, 0, stdout + "\n", "")


def test_blender_launch_is_hardened(
    box_mesh_path: Path, tmp_path: Path, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeBlender()
    monkeypatch.setattr(br, "run_command", fake)
    result = br.render_with_blender(
        box_mesh_path,
        tmp_path / "out" / "r.png",
        settings,
        br.RenderOptions(color="#ff8800", samples=8),
        blender_executable=Path("blender"),
    )
    assert result.backend == "blender" and result.renderer_version == "4.2.0"
    assert result.files == [str((tmp_path / "out" / "r.png").resolve())]
    for flag in ("--background", "--factory-startup", "--disable-autoexec"):
        assert flag in fake.args
    assert fake.args[fake.args.index("--python") + 1].endswith("_blender_job.py")
    assert fake.job["color"] == "#ff8800" and fake.job["samples"] == 8
    assert Path(fake.job["mesh"]) == box_mesh_path.resolve()


def test_turntable_and_overwrite(
    box_mesh_path: Path, tmp_path: Path, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(br, "run_command", FakeBlender())
    opts = br.RenderOptions(turntable_frames=3)
    result = br.render_with_blender(
        box_mesh_path, tmp_path / "t.png", settings, opts, blender_executable=Path("b")
    )
    assert [Path(f).name for f in result.files] == ["t_0000.png", "t_0001.png", "t_0002.png"]
    with pytest.raises(InputValidationError):
        br.render_with_blender(
            box_mesh_path, tmp_path / "t.png", settings, opts, blender_executable=Path("b")
        )


def test_off_is_converted_for_blender(
    tmp_path: Path, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    import trimesh

    off = tmp_path / "m.off"
    trimesh.creation.box().export(off)
    fake = FakeBlender()
    monkeypatch.setattr(br, "run_command", fake)
    br.render_with_blender(off, tmp_path / "r.png", settings, blender_executable=Path("b"))
    assert fake.job["mesh"].endswith("mesh.ply")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"color": "red"},
        {"color": "#12345"},
        {"background": "#zzzzzz"},
        {"width": 10},
        {"samples": 0},
        {"engine": "octane"},
        {"turntable_frames": 1000},
        {"roughness": 2},
    ],
)
def test_option_validation(kwargs: dict[str, Any]) -> None:
    with pytest.raises(InputValidationError):
        br.RenderOptions(**kwargs).validate()


def test_auto_backend_reports_missing_renderers(
    box_mesh_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PATH", "")
    settings = Settings(workspace=tmp_path)
    try:
        import pyvista  # noqa: F401
    except ImportError:
        with pytest.raises(DependencyMissingError):
            br.render_mesh(box_mesh_path, tmp_path / "r.png", settings)
    else:  # pragma: no cover - depends on optional extra
        pytest.skip("pyvista installed; fallback would render")
