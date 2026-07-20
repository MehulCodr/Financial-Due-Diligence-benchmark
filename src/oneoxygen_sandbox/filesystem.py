"""Filesystem boundary enforcement for inputs and approved outputs."""

from __future__ import annotations

import hashlib
import os
import shutil
import stat
import uuid
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from oneoxygen_sandbox.errors import (
    OutputSizeExceededError,
    PathSafetyError,
    PathTraversalError,
    SymlinkRejectedError,
    ToolFailure,
)
from oneoxygen_sandbox.models import ArtifactMetadata, InputAsset, ToolErrorCode


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _reject_symlink_components(path: Path, root: Path) -> None:
    current = root
    relative = path.relative_to(root)
    for part in relative.parts:
        current = current / part
        try:
            if current.is_symlink():
                raise SymlinkRejectedError(f"symbolic links are not permitted: {current}")
        except OSError as exc:
            raise PathSafetyError(f"cannot inspect path component {current}: {exc}") from exc


def _copy_file(source: Path, destination: Path) -> None:
    if source.is_symlink():
        raise SymlinkRejectedError(f"symbolic links are not permitted in task inputs: {source}")
    mode = source.stat(follow_symlinks=False).st_mode
    if not stat.S_ISREG(mode):
        raise PathSafetyError(f"task input is not a regular file: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination, follow_symlinks=False)


def _copy_directory(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    with os.scandir(source) as entries:
        for entry in entries:
            entry_source = Path(entry.path)
            entry_destination = destination / entry.name
            if entry.is_symlink():
                raise SymlinkRejectedError(
                    f"symbolic links are not permitted in task inputs: {entry_source}"
                )
            if entry.is_dir(follow_symlinks=False):
                _copy_directory(entry_source, entry_destination)
            elif entry.is_file(follow_symlinks=False):
                _copy_file(entry_source, entry_destination)
            else:
                raise PathSafetyError(f"unsupported task input type: {entry_source}")


def _copy_input_assets(
    task_directory: Path, workspace: Path, assets: tuple[InputAsset, ...]
) -> None:
    task_root = task_directory.resolve(strict=True)
    workspace_root = workspace.resolve(strict=True)
    for asset in assets:
        source_candidate = task_root / Path(asset.source)
        _reject_symlink_components(source_candidate, task_root)
        try:
            source = source_candidate.resolve(strict=True)
        except OSError as exc:
            raise PathSafetyError(f"input asset does not exist: {asset.source}") from exc
        if not _inside(source, task_root):
            raise PathTraversalError(f"input asset escapes the task directory: {asset.source}")

        destination_relative = PurePosixPath(asset.destination)
        destination = workspace_root.joinpath(*destination_relative.parts)
        if not _inside(destination.resolve(strict=False), workspace_root):
            raise PathTraversalError(
                f"input destination escapes the workspace: {asset.destination}"
            )
        if destination.exists():
            raise PathSafetyError(f"duplicate input destination: {asset.destination}")
        if source.is_dir():
            _copy_directory(source, destination)
        else:
            _copy_file(source, destination)


def copy_input_assets(
    task_directory: Path, workspace: Path, assets: tuple[InputAsset, ...]
) -> None:
    try:
        _copy_input_assets(task_directory, workspace, assets)
    except OSError as exc:
        raise PathSafetyError(f"cannot safely copy task inputs: {exc}") from exc


def _approved_files(output_root: Path) -> list[tuple[Path, Path]]:
    if output_root.is_symlink():
        raise SymlinkRejectedError(f"output directory may not be a symbolic link: {output_root}")
    if not output_root.exists():
        return []
    if not output_root.is_dir():
        raise PathSafetyError(f"configured output path is not a directory: {output_root}")

    root = output_root.resolve(strict=True)
    approved: list[tuple[Path, Path]] = []
    stack = [output_root]
    while stack:
        directory = stack.pop()
        with os.scandir(directory) as entries:
            for entry in entries:
                path = Path(entry.path)
                if entry.is_symlink():
                    raise SymlinkRejectedError(f"output symbolic link rejected: {path}")
                resolved = path.resolve(strict=True)
                if not _inside(resolved, root):
                    raise PathTraversalError(f"output path escapes approved directory: {path}")
                if entry.is_dir(follow_symlinks=False):
                    stack.append(path)
                elif entry.is_file(follow_symlinks=False):
                    approved.append((path, resolved.relative_to(root)))
                else:
                    raise PathSafetyError(f"unsupported output artifact type: {path}")
    return sorted(approved, key=lambda item: item[1].as_posix())


def _collect_output_artifacts(
    output_root: Path,
    artifact_directory: Path,
    maximum_size_bytes: int,
) -> list[ArtifactMetadata]:
    files = _approved_files(output_root)
    total_size = 0
    sized_files: list[tuple[Path, Path, int]] = []
    for source, relative in files:
        size = source.stat(follow_symlinks=False).st_size
        total_size += size
        if total_size > maximum_size_bytes:
            raise OutputSizeExceededError(
                f"artifacts total {total_size} bytes, exceeding limit of {maximum_size_bytes} bytes"
            )
        sized_files.append((source, relative, size))

    artifact_directory.mkdir(parents=True, exist_ok=True)
    metadata: list[ArtifactMetadata] = []
    for source, relative, size in sized_files:
        destination = artifact_directory.joinpath(*relative.parts)
        destination.parent.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256()
        with source.open("rb") as input_stream, destination.open("xb") as output_stream:
            while chunk := input_stream.read(1024 * 1024):
                digest.update(chunk)
                output_stream.write(chunk)
        metadata.append(
            ArtifactMetadata(
                relative_path=relative.as_posix(),
                size_bytes=size,
                sha256=digest.hexdigest(),
            )
        )
    return metadata


def collect_output_artifacts(
    output_root: Path,
    artifact_directory: Path,
    maximum_size_bytes: int,
) -> list[ArtifactMetadata]:
    try:
        return _collect_output_artifacts(output_root, artifact_directory, maximum_size_bytes)
    except OSError as exc:
        raise PathSafetyError(f"cannot safely collect output artifacts: {exc}") from exc


@dataclass(frozen=True)
class WorkspaceReadResult:
    path: str
    requested_start_line: int
    requested_end_line: int | None
    lines: list[dict[str, str | int]]
    total_line_count: int | None
    truncated: bool


class SecureWorkspace:
    """Workspace-only filesystem view for model-facing tools."""

    def __init__(
        self,
        root: Path,
        output_relative_path: PurePosixPath,
        protected_paths: tuple[str, ...],
    ) -> None:
        self.root = root.resolve(strict=True)
        self.output_relative_path = output_relative_path
        self.protected_paths = tuple(PurePosixPath(value) for value in protected_paths)

    def normalize_model_path(self, value: str, *, allow_root: bool = False) -> PurePosixPath:
        if "\x00" in value or "\\" in value:
            self._fail(ToolErrorCode.PATH_NOT_ALLOWED, "path must be a relative POSIX path")
        if re_match_windows_drive(value):
            self._fail(ToolErrorCode.PATH_NOT_ALLOWED, "absolute paths are not allowed")
        path = PurePosixPath(value)
        if path.is_absolute():
            self._fail(ToolErrorCode.PATH_NOT_ALLOWED, "absolute paths are not allowed")
        parts = tuple(part for part in path.parts if part != ".")
        if any(part in {"", ".."} for part in parts):
            self._fail(ToolErrorCode.PATH_NOT_ALLOWED, "path traversal is not allowed")
        if not parts and not allow_root:
            self._fail(ToolErrorCode.PATH_NOT_ALLOWED, "path must not be empty")
        return PurePosixPath(*parts) if parts else PurePosixPath(".")

    def list_files(
        self, directory: str, max_depth: int, max_entries: int
    ) -> tuple[list[dict], bool]:
        relative = self.normalize_model_path(directory, allow_root=True)
        self._reject_protected(relative)
        root = self._existing_path(relative)
        if root.is_symlink():
            self._fail(ToolErrorCode.PATH_NOT_ALLOWED, "symbolic links are not allowed")
        if not root.is_dir():
            self._fail(ToolErrorCode.PATH_NOT_ALLOWED, "path is not a directory")

        entries: list[dict] = []
        truncated = False
        stack: list[tuple[Path, PurePosixPath, int]] = [(root, relative, 0)]
        while stack:
            current_host, current_relative, depth = stack.pop()
            child_entries: list[tuple[str, Path, PurePosixPath]] = []
            with os.scandir(current_host) as iterator:
                for entry in iterator:
                    child_relative = self._join_relative(current_relative, entry.name)
                    if self._is_protected(child_relative):
                        continue
                    child_entries.append((entry.name, Path(entry.path), child_relative))
            for _name, child_host, child_relative in sorted(
                child_entries, key=lambda item: item[2].as_posix()
            ):
                if len(entries) >= max_entries:
                    truncated = True
                    return entries, truncated
                entry_type = self._entry_type(child_host)
                record = {
                    "path": child_relative.as_posix(),
                    "type": entry_type,
                }
                if entry_type == "file":
                    record["size_bytes"] = child_host.stat(follow_symlinks=False).st_size
                entries.append(record)
                if entry_type == "directory" and depth < max_depth:
                    stack.append((child_host, child_relative, depth + 1))
            stack.sort(key=lambda item: item[1].as_posix(), reverse=True)
        return entries, truncated

    def read_text_file(
        self,
        path: str,
        *,
        start_line: int,
        end_line: int | None,
        max_read_size: int,
    ) -> WorkspaceReadResult:
        relative = self.normalize_model_path(path)
        self._reject_protected(relative)
        host_path = self._existing_path(relative)
        if not host_path.is_file():
            self._fail(ToolErrorCode.PATH_NOT_ALLOWED, "path is not a regular file")
        size = host_path.stat(follow_symlinks=False).st_size
        truncated = size > max_read_size
        with host_path.open("rb") as stream:
            data = stream.read(max_read_size + 1)
        if b"\0" in data:
            self._fail(ToolErrorCode.BINARY_FILE, "binary files cannot be read as text")
        if len(data) > max_read_size:
            data = data[:max_read_size]
            truncated = True
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ToolFailure(
                ToolErrorCode.BINARY_FILE.value,
                "file is not valid UTF-8 text",
            ) from exc

        lines = text.splitlines()
        total_line_count = len(lines) if not truncated else None
        selected_end = end_line if end_line is not None else len(lines)
        selected = lines[start_line - 1 : selected_end]
        numbered = [
            {"line": number, "text": line} for number, line in enumerate(selected, start=start_line)
        ]
        return WorkspaceReadResult(
            path=relative.as_posix(),
            requested_start_line=start_line,
            requested_end_line=end_line,
            lines=numbered,
            total_line_count=total_line_count,
            truncated=truncated,
        )

    def write_text_file(
        self,
        path: str,
        content: str,
        *,
        overwrite: bool,
        create_parents: bool,
        max_write_size: int,
    ) -> ArtifactMetadata:
        data = content.encode("utf-8")
        if len(data) > max_write_size:
            self._fail(ToolErrorCode.SIZE_LIMIT_EXCEEDED, "write size limit exceeded")
        relative = self.normalize_model_path(path)
        self._reject_protected(relative)
        host_path = self._path_for_write(relative, create_parents=create_parents)
        if host_path.exists() and not overwrite:
            self._fail(ToolErrorCode.PATH_NOT_ALLOWED, "file exists and overwrite is disabled")
        digest = self._atomic_write(host_path, data)
        return ArtifactMetadata(
            relative_path=relative.as_posix(),
            size_bytes=len(data),
            sha256=digest,
        )

    def replace_text(
        self,
        path: str,
        old_text: str,
        replacement_text: str,
        *,
        expected_replacements: int,
        max_read_size: int,
        max_write_size: int,
    ) -> tuple[int, ArtifactMetadata]:
        relative = self.normalize_model_path(path)
        self._reject_protected(relative)
        host_path = self._existing_path(relative)
        if not host_path.is_file():
            self._fail(ToolErrorCode.PATH_NOT_ALLOWED, "path is not a regular file")
        if host_path.stat(follow_symlinks=False).st_size > max_read_size:
            self._fail(ToolErrorCode.SIZE_LIMIT_EXCEEDED, "file size limit exceeded")
        data = host_path.read_bytes()
        if b"\0" in data:
            self._fail(ToolErrorCode.BINARY_FILE, "binary files cannot be modified as text")
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ToolFailure(
                ToolErrorCode.BINARY_FILE.value,
                "file is not valid UTF-8 text",
            ) from exc
        actual = text.count(old_text)
        if actual != expected_replacements:
            self._fail(
                ToolErrorCode.INVALID_ARGUMENTS,
                "replacement count did not match expected count",
                content={
                    "actual_replacements": actual,
                    "expected_replacements": expected_replacements,
                },
            )
        replacement = text.replace(old_text, replacement_text)
        replacement_data = replacement.encode("utf-8")
        if len(replacement_data) > max_write_size:
            self._fail(ToolErrorCode.SIZE_LIMIT_EXCEEDED, "write size limit exceeded")
        digest = self._atomic_write(host_path, replacement_data)
        return actual, ArtifactMetadata(
            relative_path=relative.as_posix(),
            size_bytes=len(replacement_data),
            sha256=digest,
        )

    def validate_submitted_artifacts(
        self,
        artifact_paths: tuple[str, ...],
        maximum_total_size_bytes: int,
    ) -> tuple[ArtifactMetadata, ...]:
        seen: set[str] = set()
        metadata: list[ArtifactMetadata] = []
        total_size = 0
        for artifact_path in artifact_paths:
            relative = self.normalize_model_path(artifact_path)
            normalized = relative.as_posix()
            if normalized in seen:
                self._fail(ToolErrorCode.INVALID_ARGUMENTS, "duplicate artifact path")
            seen.add(normalized)
            if not self._is_under(relative, self.output_relative_path):
                self._fail(
                    ToolErrorCode.PATH_NOT_ALLOWED,
                    "submitted artifacts must be inside the configured output directory",
                )
            host_path = self._existing_path(relative)
            if not host_path.is_file():
                self._fail(ToolErrorCode.PATH_NOT_ALLOWED, "artifact is not a regular file")
            size = host_path.stat(follow_symlinks=False).st_size
            total_size += size
            if total_size > maximum_total_size_bytes:
                self._fail(ToolErrorCode.SIZE_LIMIT_EXCEEDED, "artifact size limit exceeded")
            digest = hashlib.sha256(host_path.read_bytes()).hexdigest()
            metadata.append(
                ArtifactMetadata(
                    relative_path=relative.relative_to(self.output_relative_path).as_posix(),
                    size_bytes=size,
                    sha256=digest,
                )
            )
        return tuple(metadata)

    def create_runtime_file(self, content: str, max_write_size: int) -> PurePosixPath:
        data = content.encode("utf-8")
        if len(data) > max_write_size:
            self._fail(ToolErrorCode.SIZE_LIMIT_EXCEEDED, "Python source size limit exceeded")
        runtime_relative = PurePosixPath(".oneoxygen/tool-runtime")
        self._ensure_directory_without_symlinks(runtime_relative, create=True)
        runtime_path = runtime_relative / f"tool-{uuid.uuid4().hex}.py"
        host_path = self.root.joinpath(*runtime_path.parts)
        self._atomic_write(host_path, data)
        return runtime_path

    def remove_runtime_file(self, relative_path: PurePosixPath) -> None:
        if not self._is_under(relative_path, PurePosixPath(".oneoxygen/tool-runtime")):
            return
        host_path = self.root.joinpath(*relative_path.parts)
        with suppress(OSError):
            if host_path.exists() and not host_path.is_symlink():
                host_path.unlink()

    def container_path(self, relative_path: PurePosixPath) -> str:
        return f"/workspace/{relative_path.as_posix()}"

    def container_working_directory(self, directory: str) -> str:
        relative = self.normalize_model_path(directory, allow_root=True)
        self._reject_protected(relative)
        host_path = self._existing_path(relative)
        if not host_path.is_dir():
            self._fail(ToolErrorCode.PATH_NOT_ALLOWED, "working directory is not a directory")
        return "/workspace" if not relative.parts else self.container_path(relative)

    def _path_for_write(self, relative: PurePosixPath, *, create_parents: bool) -> Path:
        host_path = self.root.joinpath(*relative.parts)
        self._ensure_directory_without_symlinks(relative.parent, create=create_parents)
        if host_path.is_symlink():
            self._fail(ToolErrorCode.PATH_NOT_ALLOWED, "symbolic links are not allowed")
        if host_path.exists() and not host_path.is_file():
            self._fail(ToolErrorCode.PATH_NOT_ALLOWED, "path is not a regular file")
        resolved = host_path.resolve(strict=False)
        if not _inside(resolved, self.root):
            self._fail(ToolErrorCode.PATH_NOT_ALLOWED, "path escapes the workspace")
        return host_path

    def _ensure_directory_without_symlinks(self, relative: PurePosixPath, *, create: bool) -> None:
        current = self.root
        for part in relative.parts:
            if part == ".":
                continue
            current = current / part
            if current.is_symlink():
                self._fail(ToolErrorCode.PATH_NOT_ALLOWED, "symbolic links are not allowed")
            if not current.exists():
                if not create:
                    self._fail(ToolErrorCode.FILE_NOT_FOUND, "parent directory does not exist")
                current.mkdir()
            if not current.is_dir():
                self._fail(ToolErrorCode.PATH_NOT_ALLOWED, "parent path is not a directory")

    def _existing_path(self, relative: PurePosixPath) -> Path:
        host_path = self.root if not relative.parts else self.root.joinpath(*relative.parts)
        self._reject_symlink_components(relative, include_final=True)
        if not host_path.exists():
            self._fail(ToolErrorCode.FILE_NOT_FOUND, "file or directory was not found")
        resolved = host_path.resolve(strict=True)
        if not _inside(resolved, self.root):
            self._fail(ToolErrorCode.PATH_NOT_ALLOWED, "path escapes the workspace")
        return host_path

    def _reject_symlink_components(self, relative: PurePosixPath, *, include_final: bool) -> None:
        parts = relative.parts if include_final else relative.parts[:-1]
        current = self.root
        for part in parts:
            if part == ".":
                continue
            current = current / part
            if current.is_symlink():
                self._fail(ToolErrorCode.PATH_NOT_ALLOWED, "symbolic links are not allowed")

    def _atomic_write(self, host_path: Path, data: bytes) -> str:
        temporary = host_path.parent / f".oneoxygen-write-{uuid.uuid4().hex}.tmp"
        digest = hashlib.sha256(data).hexdigest()
        try:
            with temporary.open("xb") as stream:
                stream.write(data)
            os.replace(temporary, host_path)
        finally:
            with suppress(OSError):
                temporary.unlink()
        return digest

    def _join_relative(self, base: PurePosixPath, name: str) -> PurePosixPath:
        if not base.parts:
            return PurePosixPath(name)
        return base / name

    def _entry_type(self, path: Path) -> str:
        if path.is_symlink():
            return "symlink"
        if path.is_dir():
            return "directory"
        if path.is_file():
            return "file"
        return "other"

    def _reject_protected(self, relative: PurePosixPath) -> None:
        if self._is_protected(relative):
            self._fail(ToolErrorCode.PATH_NOT_ALLOWED, "path is protected")

    def _is_protected(self, relative: PurePosixPath) -> bool:
        return any(self._is_under(relative, protected) for protected in self.protected_paths)

    def _is_under(self, relative: PurePosixPath, parent: PurePosixPath) -> bool:
        if not parent.parts:
            return True
        try:
            relative.relative_to(parent)
        except ValueError:
            return False
        return True

    def _fail(
        self,
        code: ToolErrorCode,
        message: str,
        *,
        content: dict | None = None,
    ) -> None:
        raise ToolFailure(code.value, message, content=content)


def re_match_windows_drive(value: str) -> bool:
    return len(value) >= 2 and value[1] == ":" and value[0].isalpha()
