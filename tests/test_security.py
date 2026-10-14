from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from reverse3d.errors import (
    DependencyMissingError,
    ExternalToolError,
    InputValidationError,
    SecurityViolationError,
)
from reverse3d.security import (
    IMAGE_EXTENSIONS,
    prepare_output_file,
    resolve_executable,
    resolve_in_workspace,
    run_command,
    validate_input_file,
)

PYTHON = Path(sys.executable)


class TestWorkspaceSandbox:
    def test_relative_path_resolves_inside(self, tmp_path: Path) -> None:
        assert resolve_in_workspace("a/b.png", tmp_path) == (tmp_path / "a" / "b.png").resolve()

    def test_workspace_root_itself_is_allowed(self, tmp_path: Path) -> None:
        assert resolve_in_workspace(".", tmp_path) == tmp_path.resolve()

    @pytest.mark.parametrize("evil", ["../secret.txt", "a/../../secret.txt", "..\\..\\x"])
    def test_traversal_is_rejected(self, tmp_path: Path, evil: str) -> None:
        ws = tmp_path / "ws"
        ws.mkdir()
        with pytest.raises(SecurityViolationError):
            resolve_in_workspace(evil, ws)

    def test_absolute_path_outside_is_rejected(self, tmp_path: Path) -> None:
        ws = tmp_path / "ws"
        ws.mkdir()
        with pytest.raises(SecurityViolationError):
            resolve_in_workspace(tmp_path / "other" / "f.stl", ws)

    def test_sibling_prefix_is_rejected(self, tmp_path: Path) -> None:
        # "/ws-evil" starts with "/ws" as a string but is not inside it.
        ws = tmp_path / "ws"
        ws.mkdir()
        with pytest.raises(SecurityViolationError):
            resolve_in_workspace(tmp_path / "ws-evil" / "f.stl", ws)

    def test_symlink_escape_is_rejected(self, tmp_path: Path) -> None:
        ws = tmp_path / "ws"
        ws.mkdir()
        outside = tmp_path / "outside"
        outside.mkdir()
        link = ws / "link"
        try:
            link.symlink_to(outside, target_is_directory=True)
        except OSError:
            pytest.skip("symlinks not permitted on this system")
        with pytest.raises(SecurityViolationError):
            resolve_in_workspace("link/file.stl", ws)


class TestInputValidation:
    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(InputValidationError):
            validate_input_file(
                tmp_path / "nope.png", allowed_extensions=IMAGE_EXTENSIONS, max_bytes=10
            )

    def test_wrong_extension(self, tmp_path: Path) -> None:
        f = tmp_path / "x.exe"
        f.write_bytes(b"MZ")
        with pytest.raises(InputValidationError):
            validate_input_file(f, allowed_extensions=IMAGE_EXTENSIONS, max_bytes=100)

    def test_empty_file(self, tmp_path: Path) -> None:
        f = tmp_path / "x.png"
        f.write_bytes(b"")
        with pytest.raises(InputValidationError):
            validate_input_file(f, allowed_extensions=IMAGE_EXTENSIONS, max_bytes=100)

    def test_too_large(self, tmp_path: Path) -> None:
        f = tmp_path / "x.png"
        f.write_bytes(b"0" * 101)
        with pytest.raises(InputValidationError):
            validate_input_file(f, allowed_extensions=IMAGE_EXTENSIONS, max_bytes=100)

    def test_ok_and_case_insensitive(self, tmp_path: Path) -> None:
        f = tmp_path / "x.PNG"
        f.write_bytes(b"0" * 10)
        assert (
            validate_input_file(f, allowed_extensions=IMAGE_EXTENSIONS, max_bytes=100)
            == f.resolve()
        )


class TestOutputs:
    def test_refuses_overwrite_by_default(self, tmp_path: Path) -> None:
        f = tmp_path / "out.stl"
        f.write_text("x")
        with pytest.raises(InputValidationError):
            prepare_output_file(f, allowed_extensions={".stl"})
        assert prepare_output_file(f, allowed_extensions={".stl"}, overwrite=True) == f.resolve()

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        out = prepare_output_file(tmp_path / "a" / "b" / "m.obj", allowed_extensions={".obj"})
        assert out.parent.is_dir()

    def test_rejects_extension(self, tmp_path: Path) -> None:
        with pytest.raises(InputValidationError):
            prepare_output_file(tmp_path / "m.py", allowed_extensions={".obj"})


class TestRunCommand:
    def test_success_captures_stdout(self) -> None:
        proc = run_command(PYTHON, ["-c", "print('hola')"], timeout_s=30)
        assert proc.stdout.strip() == "hola"

    def test_arguments_are_not_shell_interpreted(self) -> None:
        payload = "a; echo pwned && whoami | cat"
        proc = run_command(PYTHON, ["-c", "import sys; print(sys.argv[1])", payload], timeout_s=30)
        assert proc.stdout.strip() == payload

    def test_nonzero_exit_raises_with_tail(self) -> None:
        with pytest.raises(ExternalToolError) as info:
            run_command(
                PYTHON, ["-c", "import sys; sys.stderr.write('boom'); sys.exit(3)"], timeout_s=30
            )
        assert info.value.returncode == 3
        assert "boom" in info.value.stderr_tail

    def test_allow_nonzero(self) -> None:
        proc = run_command(
            PYTHON, ["-c", "import sys; sys.exit(1)"], timeout_s=30, allow_nonzero=True
        )
        assert proc.returncode == 1

    def test_timeout(self) -> None:
        with pytest.raises(ExternalToolError, match="timed out"):
            run_command(PYTHON, ["-c", "import time; time.sleep(10)"], timeout_s=1)

    def test_secrets_are_scrubbed_from_child_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "should-not-leak")
        monkeypatch.setenv("MY_DB_PASSWORD", "nope")
        monkeypatch.setenv("REVERSE3D_HARMLESS", "ok")
        code = (
            "import os; g = os.environ.get; "
            "print(g('ANTHROPIC_API_KEY'), g('MY_DB_PASSWORD'), g('REVERSE3D_HARMLESS'))"
        )
        proc = run_command(PYTHON, ["-c", code], timeout_s=30)
        assert proc.stdout.split() == ["None", "None", "ok"]

    @pytest.mark.skipif(sys.platform != "win32", reason="Windows batch-file specific")
    def test_batch_metacharacters_rejected(self, tmp_path: Path) -> None:
        bat = tmp_path / "tool.bat"
        bat.write_text("@echo off\r\necho %*\r\n")
        with pytest.raises(SecurityViolationError):
            run_command(bat, ["x & calc.exe"], timeout_s=10)


class TestResolveExecutable:
    def test_override_must_exist(self, tmp_path: Path) -> None:
        with pytest.raises(DependencyMissingError):
            resolve_executable(["x"], tmp_path / "missing.exe", tool="blender")

    def test_override_used(self) -> None:
        assert resolve_executable(["nope"], PYTHON, tool="blender") == PYTHON.resolve()

    def test_missing_on_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PATH", os.devnull)
        with pytest.raises(DependencyMissingError):
            resolve_executable(["definitely-not-a-real-binary-r3d"], None, tool="colmap")
