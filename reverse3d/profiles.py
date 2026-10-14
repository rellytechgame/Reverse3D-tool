"""Performance profiles: one knob that trades speed for quality everywhere.

``fast`` is for previews and weak machines, ``balanced`` is the default and
``quality`` is for final deliverables. Explicit tool arguments always win over
the profile.
"""

from __future__ import annotations

from dataclasses import dataclass

from reverse3d.config import Settings


@dataclass(frozen=True)
class ProfileValues:
    photo_quality: str
    render_samples: int
    render_width: int
    render_height: int
    depth_stride: int
    fit_samples: int


PROFILES: dict[str, ProfileValues] = {
    "fast": ProfileValues("low", 16, 960, 720, 4, 8_000),
    "balanced": ProfileValues("medium", 64, 1600, 1200, 2, 20_000),
    "quality": ProfileValues("high", 256, 2400, 1800, 1, 50_000),
}


def profile_values(settings: Settings) -> ProfileValues:
    return PROFILES[settings.profile]
