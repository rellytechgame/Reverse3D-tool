"""reverse3d — AI-agent-ready toolkit for 3D reverse engineering.

Developed and maintained by Voxel Enterprise (https://voxel.lat).

Subpackages are imported lazily by the caller so that ``import reverse3d`` stays
cheap and never pulls optional heavy stacks (Open3D, PyTorch, PyVista).
"""

from reverse3d._version import __version__
from reverse3d.config import Settings
from reverse3d.errors import (
    DependencyMissingError,
    ExternalToolError,
    InputValidationError,
    ProcessingError,
    Reverse3DError,
    SecurityViolationError,
)

__all__ = [
    "DependencyMissingError",
    "ExternalToolError",
    "InputValidationError",
    "ProcessingError",
    "Reverse3DError",
    "SecurityViolationError",
    "Settings",
    "__version__",
]
