"""The authorship guard must pass on the official tree and block any attempt to
change who created and owns reverse3d."""

from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]
GUARDED = [
    "LICENSE",
    "NOTICE",
    "AUTHORS.md",
    "TRADEMARKS.md",
    "README.md",
    "pyproject.toml",
    "reverse3d/_about.py",
    "reverse3d/__init__.py",
    ".github/CODEOWNERS",
]


def _guard() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "check_authorship", ROOT / "scripts/check_authorship.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    for rel in GUARDED:
        dst = tmp_path / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / rel, dst)
    return tmp_path


def _edit(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    assert old in text, f"fixture drift: {old!r} not in {path.name}"
    path.write_text(text.replace(old, new), encoding="utf-8")


def test_official_tree_is_intact() -> None:
    assert _guard().check() == []


def test_package_exposes_author() -> None:
    import reverse3d

    assert reverse3d.__author__ == "Voxel Enterprise"
    assert reverse3d.__copyright__ == "Copyright 2026 Voxel Enterprise"


@pytest.mark.parametrize(
    ("file", "old", "new", "expected"),
    [
        ("pyproject.toml", 'name = "Voxel Enterprise"', 'name = "Impostor Inc"', "pyproject.toml"),
        ("reverse3d/_about.py", 'AUTHOR = "Voxel Enterprise"', 'AUTHOR = "Impostor"', "AUTHOR"),
        ("reverse3d/_about.py", "github.com/rellytechgame", "github.com/impostor", "OFFICIAL"),
        ("LICENSE", "Copyright 2026 Voxel Enterprise", "Copyright 2026 Impostor", "LICENSE"),
        ("NOTICE", "Voxel Enterprise", "Someone Else", "NOTICE"),
        ("README.md", "## Sobre Voxel Enterprise", "## Sobre nosotros", "README.md"),
        ("AUTHORS.md", "Copyright 2026 Voxel Enterprise", "Copyright 2026 X", "AUTHORS.md"),
        (".github/CODEOWNERS", "/LICENSE ", "/LICENSE_x ", "CODEOWNERS"),
    ],
)
def test_ownership_changes_are_blocked(
    tree: Path, file: str, old: str, new: str, expected: str
) -> None:
    _edit(tree / file, old, new)
    problems = _guard().check(tree)
    assert problems, "the guard must detect the change"
    assert any(expected in p for p in problems)


def test_whitespace_only_line_ending_change_is_tolerated(tree: Path) -> None:
    license_file = tree / "LICENSE"
    license_file.write_bytes(license_file.read_bytes().replace(b"\n", b"\r\n"))
    assert _guard().check(tree) == []


def test_deleting_files_is_blocked(tree: Path) -> None:
    (tree / "TRADEMARKS.md").unlink()
    (tree / "reverse3d" / "_about.py").unlink()
    problems = _guard().check(tree)
    assert any("TRADEMARKS.md is missing" in p for p in problems)
    assert any("_about.py is missing" in p for p in problems)
