"""Tests for batch processor."""

import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from noid_rag.batch import BatchProcessor, BatchResult, FileResult
from noid_rag.config import BatchConfig


class TestBatchProcessor:
    @pytest.fixture
    def processor(self, tmp_path):
        config = BatchConfig(
            max_retries=2,
            retry_min_wait=0.01,
            retry_max_wait=0.02,
            history_dir=str(tmp_path / "history"),
        )
        return BatchProcessor(config=config)

    @pytest.mark.asyncio
    async def test_process_files_success(self, processor, tmp_path):
        files = [tmp_path / "a.pdf", tmp_path / "b.pdf"]
        for f in files:
            f.touch()

        mock_fn = AsyncMock(return_value={"chunks_stored": 5, "document_id": "doc_1"})
        result = await processor.process(files, mock_fn)

        assert result.total == 2
        assert result.success == 2
        assert result.failed == 0
        assert len(result.files) == 2
        assert result.started_at != ""
        assert result.completed_at != ""

    @pytest.mark.asyncio
    async def test_error_isolation_one_fails_others_succeed(self, processor, tmp_path):
        files = [tmp_path / "a.pdf", tmp_path / "b.pdf"]
        for f in files:
            f.touch()

        # The failing function raises OSError (a retriable exception),
        # which will be retried max_retries=2 times before giving up.
        async def process_fn(path):
            if path.name == "a.pdf":
                raise OSError("processing failed")
            return {"chunks_stored": 3, "document_id": "doc_1"}

        result = await processor.process(files, process_fn)

        assert result.success == 1
        assert result.failed == 1
        assert result.total == 2
        failed = [f for f in result.files if f.status == "failed"]
        assert len(failed) == 1
        assert "processing failed" in failed[0].error

    @pytest.mark.asyncio
    async def test_dry_run_mode(self, processor, tmp_path):
        files = [tmp_path / "a.pdf", tmp_path / "b.pdf"]
        for f in files:
            f.touch()

        mock_fn = AsyncMock()
        result = await processor.process(files, mock_fn, dry_run=True)

        assert result.total == 2
        assert result.skipped == 2
        assert result.success == 0
        assert result.failed == 0
        # process_fn should never be called in dry run
        mock_fn.assert_not_called()

    @pytest.mark.asyncio
    async def test_history_saved_to_file(self, processor, tmp_path):
        files = [tmp_path / "a.pdf"]
        files[0].touch()

        mock_fn = AsyncMock(return_value={"chunks_stored": 1, "document_id": "doc_1"})
        result = await processor.process(files, mock_fn)

        history_file = Path(processor._history_dir) / f"{result.run_id}.json"
        assert history_file.exists()

        saved = json.loads(history_file.read_text())
        assert saved["total"] == 1
        assert saved["success"] == 1
        assert len(saved["files"]) == 1

    @pytest.mark.asyncio
    async def test_get_failed_files_returns_correct_paths(self, processor, tmp_path):
        files = [tmp_path / "good.pdf", tmp_path / "bad.pdf"]
        for f in files:
            f.touch()

        call_count = 0

        async def process_fn(path):
            nonlocal call_count
            call_count += 1
            if path.name == "bad.pdf":
                raise OSError("fail")
            return {"chunks_stored": 1, "document_id": "doc_1"}

        result = await processor.process(files, process_fn)

        failed = processor.get_failed_files(result.run_id)
        assert len(failed) == 1
        assert str(tmp_path / "bad.pdf") in failed[0]

    def test_get_failed_files_nonexistent_run(self, processor):
        assert processor.get_failed_files("nonexistent_run_id") == []

    @pytest.mark.asyncio
    async def test_progress_callback_is_called(self, processor, tmp_path):
        files = [tmp_path / "a.pdf", tmp_path / "b.pdf"]
        for f in files:
            f.touch()

        progress_calls = []

        def progress(filename, current, total):
            progress_calls.append((filename, current, total))

        mock_fn = AsyncMock(return_value={"chunks_stored": 1, "document_id": "doc_1"})
        await processor.process(files, mock_fn, progress=progress)

        assert len(progress_calls) == 2
        assert progress_calls[0] == ("a.pdf", 1, 2)
        assert progress_calls[1] == ("b.pdf", 2, 2)

    @pytest.mark.asyncio
    async def test_continue_on_error_false_stops_on_first_failure(self, tmp_path):
        config = BatchConfig(
            max_retries=1,
            retry_min_wait=0.01,
            retry_max_wait=0.02,
            continue_on_error=False,
            history_dir=str(tmp_path / "history"),
        )
        processor = BatchProcessor(config=config)

        files = [tmp_path / "a.pdf", tmp_path / "b.pdf", tmp_path / "c.pdf"]
        for f in files:
            f.touch()

        async def process_fn(path):
            if path.name == "a.pdf":
                raise OSError("first fails")
            return {"chunks_stored": 1, "document_id": "doc_2"}

        result = await processor.process(files, process_fn)

        assert result.failed == 1
        assert result.success == 0
        # Only the first file should have been processed
        assert len(result.files) == 1

    @pytest.mark.asyncio
    async def test_parallel_processing_with_concurrency(self, tmp_path):
        """With concurrency > 1, all files are processed and results collected."""
        config = BatchConfig(
            max_retries=1,
            retry_min_wait=0.01,
            retry_max_wait=0.02,
            concurrency=3,
            history_dir=str(tmp_path / "history"),
        )
        processor = BatchProcessor(config=config)

        files = [tmp_path / f"{i}.pdf" for i in range(5)]
        for f in files:
            f.touch()

        mock_fn = AsyncMock(return_value={"chunks_stored": 2, "document_id": "doc_1"})
        result = await processor.process(files, mock_fn)

        assert result.total == 5
        assert result.success == 5
        assert result.failed == 0
        assert len(result.files) == 5
        assert mock_fn.call_count == 5

    @pytest.mark.asyncio
    async def test_parallel_progress_callback_called_for_all_files(self, tmp_path):
        """Progress callback is called once per file when using concurrency > 1."""
        config = BatchConfig(
            max_retries=1,
            retry_min_wait=0.01,
            retry_max_wait=0.02,
            concurrency=2,
            history_dir=str(tmp_path / "history"),
        )
        processor = BatchProcessor(config=config)

        files = [tmp_path / f"{i}.pdf" for i in range(4)]
        for f in files:
            f.touch()

        progress_calls: list[tuple[str, int, int]] = []

        def progress(filename, current, total):
            progress_calls.append((filename, current, total))

        mock_fn = AsyncMock(return_value={"chunks_stored": 1, "document_id": "doc_1"})
        await processor.process(files, mock_fn, progress=progress)

        assert len(progress_calls) == 4
        # Every call should report total=4
        assert all(total == 4 for _, _, total in progress_calls)

    @pytest.mark.asyncio
    async def test_history_atomic_write(self, tmp_path):
        """History file should be written atomically (no partial file left on error)."""
        import json as _json
        import os

        config = BatchConfig(
            max_retries=1,
            retry_min_wait=0.01,
            retry_max_wait=0.02,
            history_dir=str(tmp_path / "history"),
        )
        processor = BatchProcessor(config=config)

        file = tmp_path / "a.pdf"
        file.touch()

        mock_fn = AsyncMock(return_value={"chunks_stored": 1, "document_id": "doc_1"})
        result = await processor.process([file], mock_fn)

        history_dir = Path(processor._history_dir)
        history_file = history_dir / f"{result.run_id}.json"
        assert history_file.exists()

        # No orphaned temp files should remain
        tmp_files = [f for f in history_dir.iterdir() if f.suffix == ".json" and f != history_file]
        assert tmp_files == [], f"Orphaned temp files found: {tmp_files}"

        # File must be valid JSON
        data = _json.loads(history_file.read_text())
        assert data["total"] == 1

        # os.replace is atomic on POSIX; verify the file is the final destination
        assert os.path.samefile(history_file, history_file)


class TestFileResult:
    def test_file_result_creation(self):
        fr = FileResult(
            path="/tmp/test.pdf",
            status="success",
            chunks_count=10,
            document_id="doc_1",
            duration_ms=150.5,
        )
        assert fr.path == "/tmp/test.pdf"
        assert fr.status == "success"
        assert fr.chunks_count == 10

    def test_file_result_defaults(self):
        fr = FileResult(path="/tmp/test.pdf", status="failed")
        assert fr.chunks_count == 0
        assert fr.document_id == ""
        assert fr.error == ""
        assert fr.duration_ms == 0


class TestBatchResult:
    def test_batch_result_defaults(self):
        br = BatchResult()
        assert br.total == 0
        assert br.success == 0
        assert br.failed == 0
        assert br.skipped == 0
        assert br.files == []
        assert br.run_id  # Should be auto-generated
