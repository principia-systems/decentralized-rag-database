"""
File locking utilities for safe concurrent access to shared files.

This module provides utilities to safely read and write the jobs.json file
when multiple workers/processes are running concurrently.
"""

import json
import time
import fcntl
import contextlib
from pathlib import Path
from typing import Dict, Any

PROJECT_ROOT = Path(__file__).parent.parent.parent


@contextlib.contextmanager
def file_lock(file_path: Path, timeout: int = 10):
    """
    Context manager for file locking using fcntl (Unix/Linux/macOS).

    Args:
        file_path: Path to the file to lock
        timeout: Maximum time to wait for lock in seconds

    Yields:
        file: Open file handle with exclusive lock

    Raises:
        TimeoutError: If unable to acquire lock within timeout
    """
    file_path.parent.mkdir(exist_ok=True)

    # Open file in read-write mode, create if it doesn't exist
    fd = open(file_path, "a+")
    fd.seek(0)

    start_time = time.time()
    while True:
        try:
            # Try to acquire exclusive lock (non-blocking)
            fcntl.flock(fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except IOError:
            # Lock not available, check timeout
            if time.time() - start_time > timeout:
                fd.close()
                raise TimeoutError(
                    f"Could not acquire lock on {file_path} within {timeout} seconds"
                )
            # Wait a short time before retrying
            time.sleep(0.1)

    try:
        yield fd
    finally:
        # Release lock and close file
        fcntl.flock(fd.fileno(), fcntl.LOCK_UN)
        fd.close()


def load_jobs_safe() -> Dict[str, Any]:
    """
    Safely load jobs from temp/jobs.json with file locking.

    Returns:
        dict: Jobs data or empty dict if file doesn't exist or is corrupted
    """
    jobs_file = PROJECT_ROOT / "temp" / "jobs.json"

    try:
        with file_lock(jobs_file) as fd:
            content = fd.read()
            if content.strip():
                fd.seek(0)
                return json.load(fd)
            else:
                return {}
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"[FILE_LOCK] Error loading jobs.json: {e}, returning empty dict")
        return {}
    except Exception as e:
        print(f"[FILE_LOCK] Unexpected error loading jobs.json: {e}")
        return {}


def save_jobs_safe(jobs_data: Dict[str, Any]) -> bool:
    """
    Safely save jobs to temp/jobs.json with file locking.

    Args:
        jobs_data: Jobs data to save

    Returns:
        bool: True if successful, False otherwise
    """
    jobs_file = PROJECT_ROOT / "temp" / "jobs.json"

    try:
        with file_lock(jobs_file) as fd:
            # Truncate file and write new data
            fd.seek(0)
            fd.truncate()
            json.dump(jobs_data, fd, indent=2)
            fd.flush()  # Ensure data is written to disk
        print("[FILE_LOCK] Successfully updated jobs.json")
        return True
    except Exception as e:
        print(f"[FILE_LOCK] Error saving jobs.json: {e}")
        return False


def increment_job_progress_safe(user_email: str, increment: int = 1) -> bool:
    """
    Safely increment job progress for a user with file locking.

    Args:
        user_email: Email of the user
        increment: Number to increment by (default: 1)

    Returns:
        bool: True if successful, False otherwise
    """
    jobs_file = PROJECT_ROOT / "temp" / "jobs.json"

    try:
        with file_lock(jobs_file) as fd:
            # Read current data
            content = fd.read()
            if content.strip():
                fd.seek(0)
                jobs = json.load(fd)
            else:
                jobs = {}

            # Update job progress
            if user_email in jobs:
                if isinstance(jobs[user_email], list) and len(jobs[user_email]) >= 2:
                    jobs[user_email][1] += increment

                    # Write updated data
                    fd.seek(0)
                    fd.truncate()
                    json.dump(jobs, fd, indent=2)
                    fd.flush()

                    print(
                        f"[FILE_LOCK] Incremented job progress for {user_email}: {jobs[user_email][1]}/{jobs[user_email][0]}"
                    )
                    return True
                else:
                    print(f"[FILE_LOCK] Invalid job structure for {user_email}")
                    return False
            else:
                print(f"[FILE_LOCK] No job found for {user_email}")
                return False

    except Exception as e:
        print(f"[FILE_LOCK] Error updating job progress for {user_email}: {e}")
        return False


def reset_job_tracking_safe(user_email: str, total_jobs: int) -> bool:
    """
    Safely reset job tracking for a user, starting fresh with new totals.

    Args:
        user_email: Email of the user
        total_jobs: Total number of jobs for this batch

    Returns:
        bool: True if successful, False otherwise
    """
    jobs_file = PROJECT_ROOT / "temp" / "jobs.json"

    try:
        with file_lock(jobs_file) as fd:
            # Read current data
            content = fd.read()
            if content.strip():
                fd.seek(0)
                jobs = json.load(fd)
            else:
                jobs = {}

            # Reset job tracking (start fresh)
            jobs[user_email] = [total_jobs, 0]  # [total_jobs, completed_jobs]

            # Write updated data
            fd.seek(0)
            fd.truncate()
            json.dump(jobs, fd, indent=2)
            fd.flush()

            print(f"[FILE_LOCK] Reset job tracking for {user_email}: 0/{total_jobs}")
            return True

    except Exception as e:
        print(f"[FILE_LOCK] Error resetting job tracking for {user_email}: {e}")
        return False


def initialize_job_tracking_safe(user_email: str, total_jobs: int) -> bool:
    """
    Safely initialize job tracking for a user with file locking.

    If user already exists, adds to the total job count while preserving completed jobs.
    If user doesn't exist, creates new tracking starting at 0 completed.

    Args:
        user_email: Email of the user
        total_jobs: Number of new jobs to add for this user

    Returns:
        bool: True if successful, False otherwise
    """
    jobs_file = PROJECT_ROOT / "temp" / "jobs.json"

    try:
        with file_lock(jobs_file) as fd:
            # Read current data
            content = fd.read()
            if content.strip():
                fd.seek(0)
                jobs = json.load(fd)
            else:
                jobs = {}

            # Handle existing vs new user
            if (
                user_email in jobs
                and isinstance(jobs[user_email], list)
                and len(jobs[user_email]) >= 2
            ):
                # User exists - add to existing total while preserving completed count
                existing_total, existing_completed = jobs[user_email]
                new_total = existing_total + total_jobs
                jobs[user_email] = [new_total, existing_completed]
                print(
                    f"[FILE_LOCK] Updated job tracking for {user_email}: {existing_completed}/{new_total} (added {total_jobs} new jobs)"
                )
            else:
                # New user - start fresh
                jobs[user_email] = [total_jobs, 0]  # [total_jobs, completed_jobs]
                print(
                    f"[FILE_LOCK] Initialized job tracking for {user_email}: 0/{total_jobs}"
                )

            # Write updated data
            fd.seek(0)
            fd.truncate()
            json.dump(jobs, fd, indent=2)
            fd.flush()

            return True

    except Exception as e:
        print(f"[FILE_LOCK] Error initializing job tracking for {user_email}: {e}")
        return False
