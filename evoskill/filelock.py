"""Cross-platform file locking for process-safe storage operations.

This module provides a unified interface for file locking that works on both
Unix (via fcntl) and Windows (via msvcrt) systems.

Usage::

    from evoskill.filelock import FileLock

    # Context manager style (recommended)
    with FileLock(Path("/path/to/file")):
        # Critical section - exclusive access to file
        pass

    # Manual style
    lock = FileLock(Path("/path/to/file"), timeout=5.0)
    if lock.acquire():
        try:
            # Critical section
            pass
        finally:
            lock.release()
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Optional, Union

# Platform-specific imports
if os.name == "nt":  # Windows
    import msvcrt
else:  # Unix/Linux/MacOS
    import fcntl


class FileLock:
    """A cross-platform file lock using OS-level primitives.

    On Unix systems, uses ``fcntl.flock()`` for advisory locking.
    On Windows, uses ``msvcrt.locking()`` for mandatory locking.

    This is a **process-level** lock, not thread-level. Multiple threads
    in the same process can hold the lock simultaneously (they share
    the file descriptor). Use ``threading.Lock`` if you need thread-level
    synchronization within a single process.

    Parameters
    ----------
    path : Path or str
        Path to the file to lock. The file will be created if it doesn't exist.
    timeout : float, optional
        Maximum seconds to wait for the lock. ``None`` means wait forever.
        Default is ``None``.
    check_interval : float
        Seconds between lock acquisition attempts when polling. Default 0.1.

    Raises
    ------
    TimeoutError
        If the lock cannot be acquired within ``timeout`` seconds.
    RuntimeError
        If releasing a lock that wasn't acquired.

    Examples
    --------
    Basic usage with context manager::

        with FileLock("/tmp/myfile.jsonl"):
            with open("/tmp/myfile.jsonl", "a") as f:
                f.write("data\\n")

    With timeout handling::

        try:
            with FileLock("/tmp/myfile.jsonl", timeout=5.0):
                # Do work
                pass
        except TimeoutError:
            print("Could not acquire lock")
    """

    def __init__(
        self,
        path: Union[Path, str],
        timeout: Optional[float] = None,
        check_interval: float = 0.1,
    ) -> None:
        self._path = Path(path)
        self._timeout = timeout
        self._check_interval = check_interval
        self._lock_file: Optional[object] = None
        self._acquired = False
        # Thread-level lock to prevent concurrent acquire/release in same process
        self._thread_lock = threading.Lock()

    def _ensure_lock_file(self) -> None:
        """Ensure the lock file exists and is open."""
        if self._lock_file is None:
            # Create parent directory if needed
            self._path.parent.mkdir(parents=True, exist_ok=True)
            # Open in append mode to create if not exists
            self._lock_file = open(self._path, "a+")

    def _lock_unix(self, exclusive: bool = True) -> None:
        """Acquire lock on Unix using fcntl."""
        self._ensure_lock_file()
        operation = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
        fcntl.flock(self._lock_file.fileno(), operation)

    def _unlock_unix(self) -> None:
        """Release lock on Unix."""
        if self._lock_file is not None:
            fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_UN)

    def _lock_windows(self, exclusive: bool = True) -> None:
        """Acquire lock on Windows using msvcrt."""
        self._ensure_lock_file()
        # Windows locking is byte-range based and mandatory
        # Lock the entire file (very large number for size)
        msvcrt.locking(self._lock_file.fileno(), msvcrt.LK_LOCK, 2**31 - 1)

    def _unlock_windows(self) -> None:
        """Release lock on Windows."""
        if self._lock_file is not None:
            # LK_UNLCK releases the lock
            msvcrt.locking(self._lock_file.fileno(), msvcrt.LK_UNLCK, 2**31 - 1)

    def _try_lock(self) -> bool:
        """Try to acquire lock without blocking. Returns True if successful."""
        try:
            if os.name == "nt":
                # Windows: try non-blocking lock first
                try:
                    self._ensure_lock_file()
                    msvcrt.locking(
                        self._lock_file.fileno(),
                        msvcrt.LK_NBLCK,  # Non-blocking
                        2**31 - 1
                    )
                    return True
                except OSError:
                    return False
            else:
                # Unix: use LOCK_NB for non-blocking
                self._ensure_lock_file()
                fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                return True
        except (IOError, OSError):
            return False

    def acquire(self, blocking: bool = True) -> bool:
        """Acquire the file lock.

        Parameters
        ----------
        blocking : bool
            If ``True``, block until lock is acquired (respecting timeout).
            If ``False``, return immediately with result.

        Returns
        -------
        bool
            ``True`` if lock was acquired, ``False`` if non-blocking and unavailable.

        Raises
        ------
        TimeoutError
            If blocking with timeout and lock cannot be acquired in time.
        """
        with self._thread_lock:
            if self._acquired:
                return True  # Already acquired in this instance

            if not blocking:
                if self._try_lock():
                    self._acquired = True
                    return True
                return False

            # Blocking acquisition with optional timeout
            start_time = time.monotonic()
            while True:
                if self._try_lock():
                    self._acquired = True
                    return True

                if self._timeout is not None:
                    elapsed = time.monotonic() - start_time
                    if elapsed >= self._timeout:
                        raise TimeoutError(
                            f"Could not acquire lock on {self._path} "
                            f"within {self._timeout} seconds"
                        )

                time.sleep(self._check_interval)

    def release(self) -> None:
        """Release the file lock.

        Raises
        ------
        RuntimeError
            If called without first acquiring the lock.
        """
        with self._thread_lock:
            if not self._acquired:
                raise RuntimeError("Cannot release a lock that wasn't acquired")

            try:
                if os.name == "nt":
                    self._unlock_windows()
                else:
                    self._unlock_unix()
            finally:
                self._acquired = False
                if self._lock_file is not None:
                    self._lock_file.close()
                    self._lock_file = None

    def __enter__(self) -> FileLock:
        """Context manager entry - acquires lock."""
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - releases lock."""
        self.release()

    @property
    def is_acquired(self) -> bool:
        """Check if this instance currently holds the lock."""
        return self._acquired
