"""Trace management routes: upload, list, view, stats, delete."""

from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, Query, UploadFile

from .. import storage
from ..models import (
    TraceFormat,
    TraceInfo,
    TraceSample,
    TraceStats,
)
from ..trace_ingest import parse_records, persist_trace_bytes

router = APIRouter(prefix="/traces", tags=["Traces"])


def _trace_info(meta: dict) -> TraceInfo:
    return TraceInfo(
        **{key: value for key, value in meta.items() if key != "stats"},
        stats=TraceStats(**meta["stats"]) if meta.get("stats") else None,
    )


def _require_trace_meta(trace_id: str) -> dict:
    meta = storage.get_trace_meta(trace_id)
    if not meta:
        raise HTTPException(404, "Trace not found")
    return meta


@router.post("", response_model=TraceInfo, summary="Upload a trace file")
async def upload_trace(
    file: UploadFile = File(...),
    fmt: TraceFormat = Query(TraceFormat.jsonl, description="File format"),
):
    content = await file.read()
    try:
        meta = persist_trace_bytes(content, fmt, file.filename or storage.new_id())
    except Exception as exc:
        raise HTTPException(422, f"Could not parse trace: {exc}")
    return _trace_info(meta)


@router.get("", response_model=list[TraceInfo], summary="List all traces")
async def list_traces():
    return [_trace_info(meta) for meta in storage.list_traces()]


@router.get("/{trace_id}", response_model=TraceInfo, summary="Get trace metadata")
async def get_trace(trace_id: str):
    return _trace_info(_require_trace_meta(trace_id))


@router.get(
    "/{trace_id}/sample", response_model=TraceSample, summary="Get sample records"
)
async def sample_trace(trace_id: str, limit: int = Query(50, ge=1, le=500)):
    meta = _require_trace_meta(trace_id)
    path = storage.trace_upload_path(trace_id)
    records = parse_records(path, meta["format"])
    return TraceSample(records=records[:limit], total=len(records))


@router.delete("/{trace_id}", summary="Delete a trace")
async def delete_trace(trace_id: str):
    if not storage.delete_trace(trace_id):
        raise HTTPException(404, "Trace not found")
    return {"deleted": trace_id}
