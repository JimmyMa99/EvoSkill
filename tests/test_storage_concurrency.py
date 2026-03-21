"""Tests for storage concurrency safety.

These tests verify that TraceStorage correctly handles concurrent access
from multiple processes using file locking and atomic writes.
"""

import multiprocessing
import os
import time
from pathlib import Path

import pytest

from evoskill.config import GlobalConfig, StorageConfig
from evoskill.filelock import FileLock
from evoskill.schema import Feedback, Message, Trace
from evoskill.storage import TraceStorage


# =============================================================================
# FileLock Tests
# =============================================================================


def test_file_lock_basic(tmp_path: Path):
    """Test basic lock acquire and release."""
    lock_file = tmp_path / "test.lock"
    lock = FileLock(lock_file)

    assert not lock.is_acquired
    lock.acquire()
    assert lock.is_acquired
    lock.release()
    assert not lock.is_acquired


def test_file_lock_context_manager(tmp_path: Path):
    """Test lock as context manager."""
    lock_file = tmp_path / "test.lock"

    with FileLock(lock_file) as lock:
        assert lock.is_acquired

    assert not lock.is_acquired


def test_file_lock_non_blocking(tmp_path: Path):
    """Test non-blocking lock acquisition."""
    lock_file = tmp_path / "test.lock"
    lock1 = FileLock(lock_file)
    lock2 = FileLock(lock_file)

    # First lock acquires successfully
    assert lock1.acquire(blocking=False)

    # Second lock should fail (non-blocking)
    assert not lock2.acquire(blocking=False)

    lock1.release()


def test_file_lock_timeout(tmp_path: Path):
    """Test lock timeout."""
    lock_file = tmp_path / "test.lock"
    lock1 = FileLock(lock_file)
    lock2 = FileLock(lock_file, timeout=0.1)

    # First lock holds the lock
    lock1.acquire()

    # Second lock should timeout
    with pytest.raises(TimeoutError):
        lock2.acquire()

    lock1.release()


def test_file_lock_release_without_acquire(tmp_path: Path):
    """Test releasing a lock that wasn't acquired raises error."""
    lock_file = tmp_path / "test.lock"
    lock = FileLock(lock_file)

    with pytest.raises(RuntimeError, match="wasn't acquired"):
        lock.release()


# =============================================================================
# Concurrency Helper Functions (run in separate processes)
# =============================================================================


def _worker_append(storage_path: Path, trace_id: str, num_writes: int) -> int:
    """Worker process: append multiple traces."""
    config = StorageConfig(trace_path=storage_path, enable_file_locking=True)
    storage = TraceStorage(config)

    for i in range(num_writes):
        trace = Trace(
            id=f"{trace_id}-{i}",
            inputs=[Message(role="user", content=f"msg-{i}")],
            prediction=Message(role="assistant", content=f"resp-{i}"),
        )
        storage.append(trace)

    return num_writes


def _worker_upsert(storage_path: Path, trace_id: str, num_updates: int) -> int:
    """Worker process: upsert the same trace multiple times."""
    config = StorageConfig(trace_path=storage_path, enable_file_locking=True)
    storage = TraceStorage(config)

    for i in range(num_updates):
        trace = Trace(
            id=trace_id,  # Same ID for all updates
            inputs=[Message(role="user", content=f"update-{i}")],
            prediction=Message(role="assistant", content=f"resp-{i}"),
            feedback=Feedback(score=0.5, critique=f"iteration-{i}"),
        )
        storage.upsert(trace)

    return num_updates


def _worker_append_no_lock(storage_path: Path, trace_id: str, num_writes: int) -> int:
    """Worker process: append without locking (for comparison)."""
    config = StorageConfig(trace_path=storage_path, enable_file_locking=False)
    storage = TraceStorage(config)

    for i in range(num_writes):
        trace = Trace(
            id=f"{trace_id}-{i}",
            inputs=[Message(role="user", content=f"msg-{i}")],
            prediction=Message(role="assistant", content=f"resp-{i}"),
        )
        storage.append(trace)

    return num_writes


# =============================================================================
# TraceStorage Concurrency Tests
# =============================================================================


@pytest.mark.slow
class TestTraceStorageConcurrency:
    """Tests that require multiprocessing. Marked as slow to skip in quick runs."""

    def test_concurrent_append(self, tmp_path: Path):
        """Multiple processes append concurrently - verify no data corruption."""
        trace_file = tmp_path / "traces.jsonl"
        num_processes = 4
        writes_per_process = 25

        # Start multiple processes that all append to the same file
        with multiprocessing.Pool(num_processes) as pool:
            results = [
                pool.apply_async(
                    _worker_append,
                    (trace_file, f"worker-{i}", writes_per_process),
                )
                for i in range(num_processes)
            ]
            # Wait for all to complete
            for r in results:
                r.get(timeout=30)

        # Verify all traces were written correctly
        config = StorageConfig(trace_path=trace_file)
        storage = TraceStorage(config)
        traces = storage.load_all()

        expected_count = num_processes * writes_per_process
        assert len(traces) == expected_count, (
            f"Expected {expected_count} traces, got {len(traces)}. "
            "This indicates data loss due to race conditions."
        )

        # Verify each trace has valid content
        for trace in traces:
            assert trace.inputs[0].role == "user"
            assert trace.prediction.role == "assistant"

    def test_concurrent_upsert(self, tmp_path: Path):
        """Multiple processes upsert the same trace - verify final consistency."""
        trace_file = tmp_path / "traces.jsonl"
        num_processes = 4
        updates_per_process = 10
        shared_trace_id = "shared-trace"

        # Pre-create the trace
        config = StorageConfig(trace_path=trace_file)
        storage = TraceStorage(config)
        initial_trace = Trace(
            id=shared_trace_id,
            inputs=[Message(role="user", content="initial")],
            prediction=Message(role="assistant", content="initial"),
        )
        storage.append(initial_trace)

        # Multiple processes upsert the same trace
        with multiprocessing.Pool(num_processes) as pool:
            results = [
                pool.apply_async(
                    _worker_upsert,
                    (trace_file, shared_trace_id, updates_per_process),
                )
                for i in range(num_processes)
            ]
            for r in results:
                r.get(timeout=30)

        # Verify there's exactly one trace with the shared ID
        traces = storage.load_all()
        shared_traces = [t for t in traces if t.id == shared_trace_id]
        assert len(shared_traces) == 1, (
            f"Expected 1 trace with shared ID, got {len(shared_traces)}. "
            "This indicates concurrent upsert lost updates."
        )

        # The trace should have feedback from one of the updates
        assert shared_traces[0].feedback is not None


class TestAtomicWrites:
    """Tests for atomic write behavior."""

    def test_atomic_write_no_partial_data(self, tmp_path: Path):
        """Verify that partial writes cannot be observed."""
        trace_file = tmp_path / "traces.jsonl"
        config = StorageConfig(trace_path=trace_file)
        storage = TraceStorage(config)

        # Create many traces
        traces = [
            Trace(
                id=f"trace-{i}",
                inputs=[Message(role="user", content=f"content-{i}")],
                prediction=Message(role="assistant", content=f"response-{i}"),
            )
            for i in range(100)
        ]

        # Write them all
        storage._write_all(traces)

        # Verify all are readable and valid
        loaded = storage.load_all()
        assert len(loaded) == 100
        for i, trace in enumerate(loaded):
            assert trace.id == f"trace-{i}"
            assert trace.inputs[0].content == f"content-{i}"

    def test_upsert_maintains_other_traces(self, tmp_path: Path):
        """Verify that upserting one trace doesn't affect others."""
        trace_file = tmp_path / "traces.jsonl"
        config = StorageConfig(trace_path=trace_file)
        storage = TraceStorage(config)

        # Create initial traces
        for i in range(5):
            trace = Trace(
                id=f"trace-{i}",
                inputs=[Message(role="user", content=f"content-{i}")],
                prediction=Message(role="assistant", content=f"response-{i}"),
            )
            storage.append(trace)

        # Upsert one trace
        updated = Trace(
            id="trace-2",
            inputs=[Message(role="user", content="updated")],
            prediction=Message(role="assistant", content="updated-response"),
            feedback=Feedback(score=0.8, critique="good"),
        )
        storage.upsert(updated)

        # Verify all 5 traces exist with correct content
        loaded = storage.load_all()
        assert len(loaded) == 5

        trace_2 = next(t for t in loaded if t.id == "trace-2")
        assert trace_2.feedback is not None
        assert trace_2.inputs[0].content == "updated"

        # Verify others are unchanged
        trace_0 = next(t for t in loaded if t.id == "trace-0")
        assert trace_0.inputs[0].content == "content-0"


class TestLockingDisabled:
    """Tests for when locking is disabled."""

    def test_no_lock_when_disabled(self, tmp_path: Path):
        """Verify storage works when locking is disabled."""
        trace_file = tmp_path / "traces.jsonl"
        config = StorageConfig(trace_path=trace_file, enable_file_locking=False)
        storage = TraceStorage(config)

        trace = Trace(
            id="test-1",
            inputs=[Message(role="user", content="hello")],
            prediction=Message(role="assistant", content="world"),
        )

        # Should work without any locking
        storage.append(trace)

        loaded = storage.load_all()
        assert len(loaded) == 1
        assert loaded[0].id == "test-1"


# =============================================================================
# Configuration Tests
# =============================================================================


def test_storage_config_defaults():
    """Test StorageConfig default values."""
    config = StorageConfig()
    assert config.enable_file_locking is True
    assert config.lock_timeout == 10.0


def test_storage_config_custom_values():
    """Test StorageConfig with custom values."""
    config = StorageConfig(
        trace_path=Path("./custom/traces.jsonl"),
        enable_file_locking=False,
        lock_timeout=5.0,
    )
    assert config.enable_file_locking is False
    assert config.lock_timeout == 5.0


def test_storage_config_from_yaml(tmp_path: Path):
    """Test that storage config can be loaded from YAML."""
    yaml_content = """
storage:
  trace_path: ./data/custom.jsonl
  enable_file_locking: false
  lock_timeout: 30.0
"""
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml_content)

    config = GlobalConfig.from_yaml(config_file)
    assert config.storage.enable_file_locking is False
    assert config.storage.lock_timeout == 30.0
    assert config.storage.trace_path == Path("./data/custom.jsonl")
