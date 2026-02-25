"""Batch processing with retry logic and history tracking."""

from __future__ import annotations

import json
import secrets
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from noid_rag.config import BatchConfig

# Exceptions worth retrying — transient I/O and network errors
RETRIABLE_EXCEPTIONS = (
    OSError,
    ConnectionError,
    TimeoutError,
    httpx.HTTPStatusError,
    httpx.ConnectError,
    httpx.TimeoutException,
)


@dataclass
class FileResult:
    """Result of processing a single file."""

    path: str
    status: str  # "success" | "failed" | "skipped"
    chunks_count: int = 0
    document_id: str = ""
    error: str = ""
    duration_ms: float = 0


@dataclass
class BatchResult:
    """Result of a batch processing run."""

    run_id: str = field(default_factory=lambda: secrets.token_hex(6))
    total: int = 0
    success: int = 0
    failed: int = 0
    skipped: int = 0
    files: list[FileResult] = field(default_factory=list)
    started_at: str = ""
    completed_at: str = ""


ProgressCallback = Callable[[str, int, int], None]  # (filename, current, total)


class BatchProcessor:
    """Process multiple files with retry and error isolation."""

    def __init__(self, config: BatchConfig | None = None):
        self.config = config or BatchConfig()
        self._history_dir = Path(self.config.history_dir).expanduser()

    async def process(
        self,
        files: list[Path],
        process_fn: Callable[[Path], Any],  # async callable that processes one file
        progress: ProgressCallback | None = None,
        dry_run: bool = False,
    ) -> BatchResult:
        """Process files with retry and error isolation."""
        result = BatchResult(
            total=len(files),
            started_at=datetime.now(timezone.utc).isoformat(),
        )

        for i, file_path in enumerate(files):
            if progress:
                progress(file_path.name, i + 1, len(files))

            if dry_run:
                result.files.append(
                    FileResult(path=str(file_path), status="skipped")
                )
                result.skipped += 1
                continue

            file_result = await self._process_one(file_path, process_fn)
            result.files.append(file_result)

            if file_result.status == "success":
                result.success += 1
            else:
                result.failed += 1
                if not self.config.continue_on_error:
                    break

        result.completed_at = datetime.now(timezone.utc).isoformat()
        self._save_history(result)
        return result

    async def _process_one(
        self, file_path: Path, process_fn: Callable[[Path], Any]
    ) -> FileResult:
        """Process a single file with retries."""
        start = time.monotonic()
        try:
            # Create a retrying wrapper
            retrying_fn = retry(
                stop=stop_after_attempt(self.config.max_retries),
                wait=wait_exponential_jitter(
                    initial=self.config.retry_min_wait,
                    max=self.config.retry_max_wait,
                ),
                retry=retry_if_exception_type(RETRIABLE_EXCEPTIONS),
                reraise=True,
            )(process_fn)

            result = await retrying_fn(file_path)
            elapsed = (time.monotonic() - start) * 1000

            # Extract info from result
            chunks_count = 0
            doc_id = ""
            if isinstance(result, dict):
                chunks_count = result.get("chunks_stored", 0)
                doc_id = result.get("document_id", "")

            return FileResult(
                path=str(file_path),
                status="success",
                chunks_count=chunks_count,
                document_id=doc_id,
                duration_ms=elapsed,
            )
        except Exception as e:
            elapsed = (time.monotonic() - start) * 1000
            return FileResult(
                path=str(file_path),
                status="failed",
                error=str(e),
                duration_ms=elapsed,
            )

    def _save_history(self, result: BatchResult) -> None:
        """Save batch result to history directory."""
        self._history_dir.mkdir(parents=True, exist_ok=True)
        path = self._history_dir / f"{result.run_id}.json"
        path.write_text(json.dumps(asdict(result), indent=2, default=str))

    def get_failed_files(self, run_id: str) -> list[str]:
        """Get list of failed file paths from a previous run."""
        path = self._history_dir / f"{run_id}.json"
        if not path.exists():
            return []
        data = json.loads(path.read_text())
        return [f["path"] for f in data.get("files", []) if f["status"] == "failed"]
