#!/usr/bin/env python3
"""Authorship guard for the official reverse3d repository.

Fails (exit code 1) if anything that states who created and owns reverse3d is
altered: LICENSE, NOTICE, package metadata, the in-code ``_about`` constants,
the README attribution, AUTHORS.md, TRADEMARKS.md or the CODEOWNERS rules that
protect them.

It runs in CI on every push and pull request and as a pre-commit hook, so
neither a contributor nor an AI agent can change the project's ownership in the
official repository without the owner's approval. To make a legitimate change,
the owner updates this file as well; CODEOWNERS requires the owner to review
it.

Note: this protects the *official* repository. Anyone may fork open-source code
and remove this guard from the fork. Impersonation there is handled by
cryptographic provenance (signed releases, attestations) and by the trademark
policy; see docs/authenticity.md.

Standalone: stdlib only, Python 3.10+, no import of the package itself.
"""

from __future__ import annotations

import ast
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

OWNER = "Voxel Enterprise"
EXPECTED_ABOUT = {
    "AUTHOR": OWNER,
    "AUTHOR_EMAIL": "info@voxel.lat",
    "AUTHOR_URL": "https://voxel.lat",
    "COPYRIGHT": "Copyright 2026 Voxel Enterprise",
    "LICENSE": "Apache-2.0",
    "OFFICIAL_REPOSITORY": "https://github.com/voxelenterprise/reverse3d",
}

# SHA-256 of the file contents with line endings normalised to LF.
PINNED_FILES = {
    "LICENSE": "08ca2cd2118f3f2179792bd62f385649d94be280a2ee8acb51781919e2ac62a9",
    "NOTICE": "5b4b7515f1cfc219ef1f0861bc69d00fdc6c8c78871448cd7fa6c71e93c15f8d",
}

REQUIRED_TEXT = {
    "pyproject.toml": [
        'authors = [{ name = "Voxel Enterprise", email = "info@voxel.lat" }]',
        'maintainers = [{ name = "Voxel Enterprise", email = "info@voxel.lat" }]',
        'license = "Apache-2.0"',
        'Homepage = "https://voxel.lat"',
        'Repository = "https://github.com/voxelenterprise/reverse3d"',
    ],
    "README.md": [
        "Desarrollado por **[Voxel Enterprise](https://voxel.lat)**",
        "## Sobre Voxel Enterprise",
    ],
    "AUTHORS.md": [
        "es propiedad de **[Voxel Enterprise](https://voxel.lat)**",
        "Copyright 2026 Voxel Enterprise",
    ],
    "TRADEMARKS.md": [
        "pertenecen a **Voxel Enterprise**",
    ],
    "reverse3d/__init__.py": [
        "from reverse3d._about import",
    ],
}

PROTECTED_BY_CODEOWNERS = [
    "/LICENSE",
    "/NOTICE",
    "/AUTHORS.md",
    "/TRADEMARKS.md",
    "/reverse3d/_about.py",
    "/scripts/check_authorship.py",
    "/pyproject.toml",
    "/.github/CODEOWNERS",
]


def _digest(path: Path) -> str:
    data = path.read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def _about_constants(path: Path) -> dict[str, str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    values: dict[str, str] = {}
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            values[node.targets[0].id] = node.value.value
    return values


def check(root: Path = ROOT) -> list[str]:
    """Return a list of violations (empty when authorship is intact)."""
    problems: list[str] = []

    for name, expected in PINNED_FILES.items():
        path = root / name
        if not path.is_file():
            problems.append(f"{name} is missing")
        elif _digest(path) != expected:
            problems.append(f"{name} was modified (SHA-256 mismatch)")

    about = root / "reverse3d" / "_about.py"
    if not about.is_file():
        problems.append("reverse3d/_about.py is missing")
    else:
        actual = _about_constants(about)
        for key, value in EXPECTED_ABOUT.items():
            if actual.get(key) != value:
                problems.append(f"reverse3d/_about.py: {key} must be {value!r}")

    for name, snippets in REQUIRED_TEXT.items():
        path = root / name
        if not path.is_file():
            problems.append(f"{name} is missing")
            continue
        text = path.read_text(encoding="utf-8")
        problems.extend(
            f"{name}: required attribution not found: {snippet!r}"
            for snippet in snippets
            if snippet not in text
        )

    codeowners = root / ".github" / "CODEOWNERS"
    if not codeowners.is_file():
        problems.append(".github/CODEOWNERS is missing")
    else:
        owned = {
            line.split()[0]
            for line in codeowners.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }
        problems.extend(
            f"CODEOWNERS must protect {path}"
            for path in PROTECTED_BY_CODEOWNERS
            if path not in owned
        )
    return problems


def main() -> int:
    problems = check()
    if problems:
        print("AUTHORSHIP GUARD — the ownership of reverse3d cannot be changed:", file=sys.stderr)
        for problem in problems:
            print(f"  ✗ {problem}", file=sys.stderr)
        print(
            "\nreverse3d is created and owned by Voxel Enterprise (https://voxel.lat).\n"
            "Only the owner may change these files (CODEOWNERS approval + update of this guard).\n"
            "Forks must keep the attribution (Apache-2.0 §4) and use another name (TRADEMARKS.md).",
            file=sys.stderr,
        )
        return 1
    print("Authorship guard: OK (reverse3d © Voxel Enterprise)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
