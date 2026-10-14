from __future__ import annotations

from pathlib import Path

import pytest

from reverse3d.config import Settings
from reverse3d.errors import InputValidationError


def test_defaults_from_empty_env(tmp_path: Path) -> None:
    s = Settings.from_env({"REVERSE3D_WORKSPACE": str(tmp_path)})
    assert s.workspace == tmp_path.resolve()
    assert s.max_input_bytes == 512 * 1024 * 1024
    assert s.log_level == "INFO"
    assert s.blender_path is None


def test_values_are_parsed(tmp_path: Path) -> None:
    s = Settings.from_env(
        {
            "REVERSE3D_WORKSPACE": str(tmp_path),
            "REVERSE3D_MAX_INPUT_MB": "10",
            "REVERSE3D_MAX_IMAGES": "7",
            "REVERSE3D_LOG_LEVEL": "debug",
            "REVERSE3D_LOG_JSON": "1",
        }
    )
    assert s.max_input_bytes == 10 * 1024 * 1024
    assert s.max_images == 7
    assert s.log_level == "DEBUG"
    assert s.log_json is True


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("REVERSE3D_MAX_INPUT_MB", "abc"),
        ("REVERSE3D_MAX_INPUT_MB", "0"),
        ("REVERSE3D_MAX_IMAGES", "999999"),
        ("REVERSE3D_TIMEOUT_S", "-5"),
        ("REVERSE3D_LOG_LEVEL", "TRACE"),
    ],
)
def test_invalid_values_fail_loudly(name: str, value: str) -> None:
    with pytest.raises(InputValidationError):
        Settings.from_env({name: value})
