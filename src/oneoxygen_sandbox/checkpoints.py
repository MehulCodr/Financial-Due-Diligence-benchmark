"""Secure immutable workspace checkpoints for suspended agent runs."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import tempfile
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

from pydantic import Field, field_validator

from oneoxygen_sandbox.errors import PathSafetyError
from oneoxygen_sandbox.models import StrictModel, is_forbidden_environment_name

CHECKPOINT_SCHEMA_VERSION = 1
DEFAULT_MAX_CHECKPOINT_BYTES = 100 * 1024 * 1024
_MANIFEST_NAME = "checkpoint.json"
_DEFAULT_EXCLUSIONS = (
    PurePosixPath(".oneoxygen/tool-runtime"),
    PurePosixPath(".oneoxygen/grader"),
    PurePosixPath(".grader"),
    PurePosixPath("hidden_grader"),
)


class CheckpointFile(StrictModel):
    relative_path: str = Field(min_length=1, max_length=1_024)
    size_bytes: int = Field(ge=0)
    sha256: str

    @field_validator("relative_path")
    @classmethod
    def validate_relative_path(cls, value: str) -> str:
        path = PurePosixPath(value)
        if (
            value != path.as_posix()
            or path.is_absolute()
            or any(part in {"", ".", ".."} for part in path.parts)
        ):
            raise ValueError("checkpoint paths must be safe relative POSIX paths")
        return value

    @field_validator("sha256")
    @classmethod
    def validate_sha256(cls, value: str) -> str:
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError("checkpoint hashes must be lowercase SHA-256 digests")
        return value


class CheckpointManifest(StrictModel):
    schema_version: int = CHECKPOINT_SCHEMA_VERSION
    run_id: str = Field(min_length=1, max_length=128)
    generation: int = Field(ge=0)
    created_at: datetime
    total_size_bytes: int = Field(ge=0)
    files: tuple[CheckpointFile, ...]


class WorkspaceCheckpoint:
    """Capture, verify, restore, and retire bounded workspace generations."""

    def __init__(
        self,
        root: Path,
        run_id: str,
        *,
        maximum_size_bytes: int = DEFAULT_MAX_CHECKPOINT_BYTES,
        keep_generations: int = 3,
        excluded_paths: tuple[str, ...] = (),
        environ: dict[str, str] | None = None,
    ) -> None:
        if maximum_size_bytes < 1:
            raise ValueError("maximum checkpoint size must be positive")
        if keep_generations < 1:
            raise ValueError("at least one checkpoint generation must be retained")
        self.root = root.resolve()
        self.run_id = run_id
        self.maximum_size_bytes = maximum_size_bytes
        self.keep_generations = keep_generations
        self.exclusions = _DEFAULT_EXCLUSIONS + tuple(
            self._safe_relative(value) for value in excluded_paths
        )
        environment = os.environ if environ is None else environ
        self._secrets = tuple(
            value.encode("utf-8")
            for name, value in environment.items()
            if is_forbidden_environment_name(name) and len(value) >= 8
        )

    def capture(self, workspace: Path, generation: int) -> CheckpointManifest:
        source = workspace.resolve(strict=True)
        if not source.is_dir() or source.is_symlink():
            raise PathSafetyError("checkpoint source must be a real workspace directory")
        self.root.mkdir(parents=True, exist_ok=True)
        destination = self.generation_path(generation)
        if destination.exists():
            raise PathSafetyError("finalized checkpoint generations are immutable")
        temporary = Path(tempfile.mkdtemp(prefix=".checkpoint-", dir=str(self.root))).resolve()
        files: list[CheckpointFile] = []
        total = 0
        try:
            payload_root = temporary / "workspace"
            payload_root.mkdir()
            for source_file, relative in self._walk_files(source):
                data = source_file.read_bytes()
                total += len(data)
                if total > self.maximum_size_bytes:
                    raise PathSafetyError("workspace checkpoint exceeds its size limit")
                if any(secret in data for secret in self._secrets):
                    raise PathSafetyError("workspace checkpoint contains a provider credential")
                target = payload_root.joinpath(*relative.parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open("xb") as stream:
                    stream.write(data)
                files.append(
                    CheckpointFile(
                        relative_path=relative.as_posix(),
                        size_bytes=len(data),
                        sha256=hashlib.sha256(data).hexdigest(),
                    )
                )
            manifest = CheckpointManifest(
                run_id=self.run_id,
                generation=generation,
                created_at=datetime.now(UTC),
                total_size_bytes=total,
                files=tuple(sorted(files, key=lambda item: item.relative_path)),
            )
            manifest_path = temporary / _MANIFEST_NAME
            manifest_path.write_text(
                json.dumps(
                    manifest.model_dump(mode="json"),
                    ensure_ascii=True,
                    separators=(",", ":"),
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            self._make_read_only(temporary)
            os.replace(temporary, destination)
            self.verify(generation)
            return manifest
        except BaseException:
            self._make_writable(temporary)
            with suppress(OSError):
                shutil.rmtree(temporary)
            raise

    def verify(self, generation: int) -> CheckpointManifest:
        directory = self.generation_path(generation)
        if directory.is_symlink() or not directory.is_dir():
            raise PathSafetyError("checkpoint generation is missing or unsafe")
        manifest_path = directory / _MANIFEST_NAME
        if manifest_path.is_symlink() or not manifest_path.is_file():
            raise PathSafetyError("checkpoint manifest is missing or unsafe")
        try:
            manifest = CheckpointManifest.model_validate_json(
                manifest_path.read_text(encoding="utf-8")
            )
        except (OSError, ValueError) as exc:
            raise PathSafetyError("checkpoint manifest is invalid") from exc
        if manifest.run_id != self.run_id or manifest.generation != generation:
            raise PathSafetyError("checkpoint identity does not match its manifest")
        actual = {
            relative.as_posix(): path
            for path, relative in self._walk_files(directory / "workspace")
        }
        expected = {item.relative_path: item for item in manifest.files}
        if set(actual) != set(expected):
            raise PathSafetyError("checkpoint file set does not match its manifest")
        total = 0
        for relative, item in expected.items():
            data = actual[relative].read_bytes()
            total += len(data)
            if len(data) != item.size_bytes or hashlib.sha256(data).hexdigest() != item.sha256:
                raise PathSafetyError("checkpoint hash verification failed")
        if total != manifest.total_size_bytes or total > self.maximum_size_bytes:
            raise PathSafetyError("checkpoint size does not match its manifest")
        return manifest

    def restore(self, generation: int, workspace: Path) -> CheckpointManifest:
        manifest = self.verify(generation)
        destination = workspace.resolve(strict=True)
        if not destination.is_dir() or destination.is_symlink():
            raise PathSafetyError("checkpoint restore target must be a real directory")
        if any(destination.iterdir()):
            raise PathSafetyError("checkpoint restore target must be empty")
        source = self.generation_path(generation) / "workspace"
        for item in manifest.files:
            relative = self._safe_relative(item.relative_path)
            source_file = source.joinpath(*relative.parts)
            target = destination.joinpath(*relative.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with source_file.open("rb") as input_stream, target.open("xb") as output_stream:
                shutil.copyfileobj(input_stream, output_stream, length=1024 * 1024)
        return manifest

    def cleanup(self) -> tuple[int, ...]:
        generations = self.list_generations()
        removed: list[int] = []
        for generation in generations[: max(0, len(generations) - self.keep_generations)]:
            target = self.generation_path(generation).resolve()
            if target.parent != self.root or not target.name.startswith("generation-"):
                raise PathSafetyError("refusing unsafe checkpoint cleanup target")
            self._make_writable(target)
            shutil.rmtree(target)
            removed.append(generation)
        return tuple(removed)

    def list_generations(self) -> tuple[int, ...]:
        if not self.root.exists():
            return ()
        values: list[int] = []
        for child in self.root.iterdir():
            if child.is_dir() and not child.is_symlink() and child.name.startswith("generation-"):
                suffix = child.name.removeprefix("generation-")
                if suffix.isdigit():
                    values.append(int(suffix))
        return tuple(sorted(values))

    def generation_path(self, generation: int) -> Path:
        if generation < 0:
            raise ValueError("checkpoint generation cannot be negative")
        return self.root / f"generation-{generation:08d}"

    def _walk_files(self, root: Path) -> list[tuple[Path, PurePosixPath]]:
        if root.is_symlink() or not root.is_dir():
            raise PathSafetyError("checkpoint tree root is unsafe")
        files: list[tuple[Path, PurePosixPath]] = []
        stack = [root]
        while stack:
            directory = stack.pop()
            with os.scandir(directory) as entries:
                for entry in entries:
                    path = Path(entry.path)
                    relative = PurePosixPath(path.relative_to(root).as_posix())
                    if self._excluded(relative):
                        continue
                    mode = entry.stat(follow_symlinks=False).st_mode
                    if entry.is_symlink():
                        raise PathSafetyError("symbolic links are forbidden in checkpoints")
                    if stat.S_ISDIR(mode):
                        stack.append(path)
                    elif stat.S_ISREG(mode):
                        files.append((path, relative))
                    else:
                        raise PathSafetyError(
                            "device files, sockets, and named pipes are forbidden in checkpoints"
                        )
        return sorted(files, key=lambda item: item[1].as_posix())

    def _excluded(self, relative: PurePosixPath) -> bool:
        return any(self._is_under(relative, excluded) for excluded in self.exclusions)

    @staticmethod
    def _is_under(relative: PurePosixPath, parent: PurePosixPath) -> bool:
        try:
            relative.relative_to(parent)
        except ValueError:
            return False
        return True

    @staticmethod
    def _safe_relative(value: str) -> PurePosixPath:
        if "\\" in value or "\x00" in value:
            raise ValueError("checkpoint paths must use relative POSIX syntax")
        path = PurePosixPath(value)
        if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
            raise ValueError("checkpoint path is unsafe")
        return path

    @staticmethod
    def _make_read_only(root: Path) -> None:
        for directory, child_directories, filenames in os.walk(root):
            for name in filenames:
                with suppress(OSError):
                    os.chmod(Path(directory) / name, stat.S_IREAD)
            for name in child_directories:
                with suppress(OSError):
                    os.chmod(Path(directory) / name, stat.S_IREAD | stat.S_IEXEC)
        with suppress(OSError):
            os.chmod(root, stat.S_IREAD | stat.S_IEXEC)

    @staticmethod
    def _make_writable(root: Path) -> None:
        if not root.exists():
            return
        for directory, child_directories, filenames in os.walk(root):
            with suppress(OSError):
                os.chmod(directory, stat.S_IREAD | stat.S_IWRITE | stat.S_IEXEC)
            for name in (*child_directories, *filenames):
                with suppress(OSError):
                    os.chmod(
                        Path(directory) / name,
                        stat.S_IREAD | stat.S_IWRITE | stat.S_IEXEC,
                    )
