"""Atomic payload files and SQLite metadata for durable batches."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import Any

from oneoxygen_sandbox.batching.models import BatchJob, BatchRequest
from oneoxygen_sandbox.errors import ConfigurationError


class BatchArtifactStore:
    """Keep bounded large request/output payloads outside SQLite."""

    def __init__(self, root: Path, *, maximum_payload_bytes: int = 200 * 1024 * 1024) -> None:
        self.root = root.resolve()
        self.maximum_payload_bytes = maximum_payload_bytes
        self.root.mkdir(parents=True, exist_ok=True)

    def write_json(self, category: str, identifier: str, value: Any) -> tuple[str, str]:
        data = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        reference = f"{category}/{identifier}.json"
        self.write_bytes(reference, data)
        return reference, hashlib.sha256(data).hexdigest()

    def write_bytes(self, reference: str, data: bytes) -> str:
        if len(data) > self.maximum_payload_bytes:
            raise ConfigurationError("batch payload exceeds its configured size limit")
        target = self.resolve(reference)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.parent / f".{target.name}.{uuid.uuid4().hex}.tmp"
        try:
            with temporary.open("xb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, target)
        finally:
            if temporary.exists():
                temporary.unlink()
        return reference

    def read_bytes(self, reference: str) -> bytes:
        target = self.resolve(reference)
        data = target.read_bytes()
        if len(data) > self.maximum_payload_bytes:
            raise ConfigurationError("stored batch payload exceeds its configured size limit")
        return data

    def read_json(self, reference: str) -> Any:
        try:
            return json.loads(self.read_bytes(reference))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ConfigurationError("stored batch payload is invalid JSON") from exc

    def resolve(self, reference: str) -> Path:
        if "\\" in reference or "\x00" in reference:
            raise ConfigurationError("batch payload reference is unsafe")
        relative = PurePosixPath(reference)
        if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
            raise ConfigurationError("batch payload reference is unsafe")
        target = self.root.joinpath(*relative.parts).resolve(strict=False)
        try:
            target.relative_to(self.root)
        except ValueError as exc:
            raise ConfigurationError("batch payload reference escapes its store") from exc
        return target


class SQLiteBatchStore:
    """Durably enforce batch/request/custom-ID uniqueness."""

    def __init__(self, path: Path) -> None:
        self.path = path.resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA synchronous = FULL")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS batch_requests (
                    internal_request_id TEXT PRIMARY KEY,
                    payload_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS batch_jobs (
                    internal_batch_id TEXT PRIMARY KEY,
                    provider_batch_id TEXT UNIQUE,
                    payload_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS batch_mappings (
                    internal_batch_id TEXT NOT NULL,
                    provider_custom_id TEXT NOT NULL,
                    internal_request_id TEXT NOT NULL,
                    PRIMARY KEY (internal_batch_id, provider_custom_id),
                    UNIQUE (internal_batch_id, internal_request_id),
                    FOREIGN KEY (internal_batch_id)
                        REFERENCES batch_jobs(internal_batch_id) ON DELETE CASCADE,
                    FOREIGN KEY (internal_request_id)
                        REFERENCES batch_requests(internal_request_id)
                );
                """
            )

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def save_request(self, request: BatchRequest) -> None:
        payload = request.model_dump_json()
        with self.transaction() as connection:
            existing = connection.execute(
                "SELECT payload_json FROM batch_requests WHERE internal_request_id = ?",
                (request.internal_request_id,),
            ).fetchone()
            if existing is not None and existing[0] != payload:
                raise ConfigurationError("batch request identity collision")
            connection.execute(
                """
                INSERT INTO batch_requests(internal_request_id, payload_json)
                VALUES (?, ?)
                ON CONFLICT(internal_request_id) DO UPDATE SET payload_json=excluded.payload_json
                """,
                (request.internal_request_id, payload),
            )

    def load_request(self, request_id: str) -> BatchRequest:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT payload_json FROM batch_requests WHERE internal_request_id = ?",
                (request_id,),
            ).fetchone()
        if row is None:
            raise ConfigurationError("batch request was not found")
        return BatchRequest.model_validate_json(row[0])

    def save_job(self, job: BatchJob) -> None:
        payload = job.model_dump_json()
        with self.transaction() as connection:
            connection.execute(
                """
                INSERT INTO batch_jobs(internal_batch_id, provider_batch_id, payload_json)
                VALUES (?, ?, ?)
                ON CONFLICT(internal_batch_id) DO UPDATE SET
                    provider_batch_id=excluded.provider_batch_id,
                    payload_json=excluded.payload_json
                """,
                (job.internal_batch_id, job.provider_batch_id, payload),
            )

    def load_job(self, batch_id: str) -> BatchJob:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT payload_json FROM batch_jobs WHERE internal_batch_id = ?",
                (batch_id,),
            ).fetchone()
        if row is None:
            raise ConfigurationError("batch job was not found")
        return BatchJob.model_validate_json(row[0])

    def list_jobs(self, *, unfinished: bool = False) -> tuple[BatchJob, ...]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT payload_json FROM batch_jobs ORDER BY internal_batch_id"
            ).fetchall()
        jobs = tuple(BatchJob.model_validate_json(row[0]) for row in rows)
        if not unfinished:
            return jobs
        terminal = {"completed", "partially_completed", "failed", "expired", "cancelled"}
        return tuple(job for job in jobs if job.state.value not in terminal)

    def save_mapping(self, batch_id: str, custom_id: str, internal_request_id: str) -> None:
        with self.transaction() as connection:
            try:
                connection.execute(
                    """
                    INSERT INTO batch_mappings(
                        internal_batch_id, provider_custom_id, internal_request_id
                    ) VALUES (?, ?, ?)
                    """,
                    (batch_id, custom_id, internal_request_id),
                )
            except sqlite3.IntegrityError as exc:
                existing = connection.execute(
                    """
                    SELECT internal_request_id FROM batch_mappings
                    WHERE internal_batch_id = ? AND provider_custom_id = ?
                    """,
                    (batch_id, custom_id),
                ).fetchone()
                if existing is None or existing[0] != internal_request_id:
                    raise ConfigurationError("duplicate batch custom ID") from exc

    def mappings(self, batch_id: str) -> dict[str, str]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT provider_custom_id, internal_request_id FROM batch_mappings
                WHERE internal_batch_id = ? ORDER BY provider_custom_id
                """,
                (batch_id,),
            ).fetchall()
        return {str(custom_id): str(request_id) for custom_id, request_id in rows}
