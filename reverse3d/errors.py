"""Exception hierarchy.

Every error raised intentionally by reverse3d derives from :class:`Reverse3DError`
and carries a stable machine-readable ``code`` so that CLI ``--json`` output and
the agent tool layer can report failures without leaking stack traces.
"""

from __future__ import annotations


class Reverse3DError(Exception):
    """Base class for all reverse3d errors."""

    code: str = "reverse3d_error"

    def __init__(self, message: str, *, hint: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.hint = hint

    def to_dict(self) -> dict[str, str]:
        data = {"code": self.code, "message": self.message}
        if self.hint:
            data["hint"] = self.hint
        return data


class InputValidationError(Reverse3DError):
    """An argument or input file is invalid (missing, wrong type, too large...)."""

    code = "invalid_input"


class SecurityViolationError(Reverse3DError):
    """A request tried to escape the sandbox or violate a security limit."""

    code = "security_violation"


class DependencyMissingError(Reverse3DError):
    """An optional Python extra or an external binary is not available."""

    code = "dependency_missing"


class ExternalToolError(Reverse3DError):
    """An external binary (COLMAP, Blender) failed or timed out."""

    code = "external_tool_failed"

    def __init__(
        self,
        message: str,
        *,
        hint: str | None = None,
        returncode: int | None = None,
        stderr_tail: str = "",
    ) -> None:
        super().__init__(message, hint=hint)
        self.returncode = returncode
        self.stderr_tail = stderr_tail

    def to_dict(self) -> dict[str, str]:
        data = super().to_dict()
        if self.returncode is not None:
            data["returncode"] = str(self.returncode)
        if self.stderr_tail:
            data["stderr_tail"] = self.stderr_tail
        return data


class ProcessingError(Reverse3DError):
    """An algorithm could not produce a valid result for the given data."""

    code = "processing_failed"


class CaptureQualityError(ProcessingError):
    """The input capture (e.g. a photo set) is not good enough to process.

    Retrying with other parameters will not help: the user has to improve the
    capture. Kept under ``processing_failed`` so existing clients see the same code.
    """
