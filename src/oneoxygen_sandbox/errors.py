"""Structured exception hierarchy for sandbox operations."""


class SandboxError(Exception):
    """Base class for expected, user-facing sandbox failures."""

    code = "sandbox_error"


class ConfigurationError(SandboxError):
    code = "configuration_error"


class PathSafetyError(SandboxError):
    code = "unsafe_path"


class PathTraversalError(PathSafetyError):
    code = "path_traversal"


class SymlinkRejectedError(PathSafetyError):
    code = "symlink_rejected"


class OutputSizeExceededError(PathSafetyError):
    code = "output_size_exceeded"


class DockerUnavailableError(SandboxError):
    code = "docker_unavailable"


class DockerOperationError(SandboxError):
    code = "docker_operation_error"


class SecurityPolicyError(DockerOperationError):
    code = "security_policy_not_enforced"


class LifecycleError(SandboxError):
    code = "invalid_lifecycle"


class SandboxTimeoutError(SandboxError):
    code = "sandbox_timeout"


class RecordPersistenceError(SandboxError):
    code = "record_persistence_error"


class CleanupError(SandboxError):
    code = "cleanup_error"


class ToolFailure(SandboxError):
    """Sanitized failure that can be returned to a model-facing tool caller."""

    code = "tool_failure"

    def __init__(
        self,
        tool_code: str,
        message: str,
        *,
        content: dict | None = None,
        metadata: dict | None = None,
        truncated: bool = False,
    ) -> None:
        super().__init__(message)
        self.tool_code = tool_code
        self.message = message
        self.content = content or {}
        self.metadata = metadata or {}
        self.truncated = truncated
