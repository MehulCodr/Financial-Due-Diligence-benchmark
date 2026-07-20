"""Deterministic tool registry."""

from __future__ import annotations

from oneoxygen_sandbox.errors import ToolFailure
from oneoxygen_sandbox.models import ToolDefinition, ToolErrorCode
from oneoxygen_sandbox.tools.base import Tool


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ToolFailure(
                ToolErrorCode.INVALID_ARGUMENTS.value,
                f"duplicate tool registered: {tool.name}",
            )
        self._tools[tool.name] = tool

    def resolve(self, name: str) -> Tool:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise ToolFailure(
                ToolErrorCode.UNKNOWN_TOOL.value,
                "requested tool is not registered",
            ) from exc

    def definitions(self) -> tuple[ToolDefinition, ...]:
        return tuple(self._tools[name].definition() for name in sorted(self._tools))

    def provider_schemas(self) -> list[dict]:
        return [definition.to_provider_dict() for definition in self.definitions()]

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._tools))


def default_tool_registry() -> ToolRegistry:
    from oneoxygen_sandbox.tools.standard import (
        ExecutePythonTool,
        ExecuteShellTool,
        ListFilesTool,
        ReadTextFileTool,
        ReplaceTextTool,
        SubmitResultTool,
        WriteTextFileTool,
    )

    registry = ToolRegistry()
    for tool in (
        ListFilesTool(),
        ReadTextFileTool(),
        WriteTextFileTool(),
        ReplaceTextTool(),
        ExecuteShellTool(),
        ExecutePythonTool(),
        SubmitResultTool(),
    ):
        registry.register(tool)
    return registry
