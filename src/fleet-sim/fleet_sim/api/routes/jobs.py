"""Simulation job management routes."""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, BackgroundTasks, HTTPException

from .. import storage
from ..models import JobOut, JobRequest, JobType
from ..runner import run_job

router = APIRouter(prefix="/jobs", tags=["Jobs"])


def _job_out(data: dict) -> JobOut:
    return JobOut(**data)


@router.post("", response_model=JobOut, summary="Submit a simulation job")
async def create_job(body: JobRequest, background_tasks: BackgroundTasks):
    # Validate: check the right params are provided
    if body.type == JobType.optimize and not body.optimize:
        raise HTTPException(422, "optimize params required for type=optimize")
    if body.type == JobType.simulate and not body.simulate:
        raise HTTPException(422, "simulate params required for type=simulate")
    if body.type == JobType.whatif and not body.whatif:
        raise HTTPException(422, "whatif params required for type=whatif")

    job_id = storage.new_id()
    data = {
        "id": job_id,
        "type": body.type.value,
        "status": "pending",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "started_at": None,
        "completed_at": None,
        "error": None,
        "request": body.model_dump(),
        "result_optimize": None,
        "result_simulate": None,
        "result_whatif": None,
    }
    storage.save_job(job_id, data)
    background_tasks.add_task(run_job, job_id, body)
    return _job_out(data)


@router.get("", response_model=list[JobOut], summary="List all jobs")
async def list_jobs():
    return [_job_out(d) for d in storage.list_jobs()]


@router.get("/{job_id}", response_model=JobOut, summary="Get job status and results")
async def get_job(job_id: str):
    data = storage.get_job(job_id)
    if not data:
        raise HTTPException(404, "Job not found")
    return _job_out(data)


@router.delete("/{job_id}", summary="Delete a job")
async def delete_job(job_id: str):
    if not storage.delete_job(job_id):
        raise HTTPException(404, "Job not found")
    return {"deleted": job_id}
