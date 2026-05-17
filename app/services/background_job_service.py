from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import threading
from typing import Any, Callable
from uuid import uuid4


@dataclass
class BackgroundJob:
    id: str
    label: str
    status: str = "queued"
    progress: int = 0
    message: str = "Queued"
    result: Any = None
    error: str = ""
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )
    finished_at: str = ""


class BackgroundJobService:
    """Small in-process background job registry for long dashboard actions."""

    def __init__(self) -> None:
        self._jobs: dict[str, BackgroundJob] = {}
        self._lock = threading.Lock()

    def start(
        self,
        label: str,
        task: Callable[[Callable[[int, str], None]], Any],
    ) -> BackgroundJob:
        job = BackgroundJob(id=uuid4().hex, label=label)
        with self._lock:
            self._jobs[job.id] = job

        def update(progress: int, message: str) -> None:
            self.update(job.id, progress=progress, message=message)

        thread = threading.Thread(
            target=self._run,
            args=(job.id, task, update),
            daemon=True,
        )
        thread.start()
        return job

    def get(self, job_id: str) -> BackgroundJob | None:
        with self._lock:
            return self._jobs.get(job_id)

    def update(
        self,
        job_id: str,
        progress: int | None = None,
        message: str | None = None,
    ) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            if progress is not None:
                job.progress = max(0, min(100, progress))
            if message is not None:
                job.message = message

    def serialize(self, job: BackgroundJob | None) -> dict[str, Any]:
        if job is None:
            return {"status": "missing", "message": "Job not found"}
        return {
            "id": job.id,
            "label": job.label,
            "status": job.status,
            "progress": job.progress,
            "message": job.message,
            "error": job.error,
            "created_at": job.created_at,
            "finished_at": job.finished_at,
            "has_result": job.result is not None,
        }

    def _run(
        self,
        job_id: str,
        task: Callable[[Callable[[int, str], None]], Any],
        update: Callable[[int, str], None],
    ) -> None:
        with self._lock:
            job = self._jobs[job_id]
            job.status = "running"
            job.progress = 1
            job.message = "Running"
        try:
            result = task(update)
            with self._lock:
                job = self._jobs[job_id]
                job.status = "completed"
                job.progress = 100
                job.message = "Completed"
                job.result = result
                job.finished_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        except Exception as exc:
            with self._lock:
                job = self._jobs[job_id]
                job.status = "failed"
                job.error = str(exc)
                job.message = "Failed"
                job.finished_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
