"""Pydantic models for the fleet-sim REST API."""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ── GPU Profiles ──────────────────────────────────────────────────────────────

class GpuProfileOut(BaseModel):
    name: str
    W_ms: float
    H_ms_per_slot: float
    chunk: int
    blk_size: int
    total_kv_blks: int
    max_slots: int
    cost_per_hr: float


# ── Traces ────────────────────────────────────────────────────────────────────

class TraceFormat(str, Enum):
    semantic_router = "semantic_router"
    jsonl = "jsonl"
    csv = "csv"


class HistogramBucket(BaseModel):
    lo: int
    hi: int
    count: int


class TraceStats(BaseModel):
    n_requests: int
    duration_s: float
    arrival_rate_rps: float
    p50_prompt_tokens: int
    p95_prompt_tokens: int
    p99_prompt_tokens: int
    p50_output_tokens: int
    p99_output_tokens: int
    p50_total_tokens: int
    p99_total_tokens: int
    routing_distribution: Dict[str, float] = Field(default_factory=dict)
    prompt_histogram: List[HistogramBucket] = Field(default_factory=list)
    output_histogram: List[HistogramBucket] = Field(default_factory=list)


class TraceInfo(BaseModel):
    id: str
    name: str
    format: TraceFormat
    upload_time: datetime
    n_requests: int
    stats: Optional[TraceStats] = None


class TraceSample(BaseModel):
    records: List[Dict[str, Any]]
    total: int


# ── Workloads (built-in CDFs) ─────────────────────────────────────────────────

class BuiltinWorkload(BaseModel):
    name: str
    description: str
    path: str
    stats: Optional[TraceStats] = None


class CdfPoint(BaseModel):
    threshold: int
    cumulative_frac: float


# ── Fleets ────────────────────────────────────────────────────────────────────

class PoolConfigIn(BaseModel):
    pool_id: str = Field(..., description="Unique pool name, e.g. 'short' or 'llama8b'")
    gpu: str = Field(..., description="GPU profile key: 'a100', 'h100', 'a10g'")
    n_gpus: int = Field(..., ge=1)
    max_ctx: int = Field(..., ge=512, description="Max context tokens this pool handles")


class FleetConfigIn(BaseModel):
    name: str
    pools: List[PoolConfigIn]
    router: str = Field("length", description="Router type: 'length', 'model', 'random', 'least_loaded', 'compress_route'")
    compress_gamma: Optional[float] = Field(None, ge=1.0, le=3.0)


class FleetConfigOut(FleetConfigIn):
    id: str
    created_at: datetime
    total_gpus: int
    estimated_cost_per_hr: float
    estimated_annual_cost_kusd: float


# ── Jobs ──────────────────────────────────────────────────────────────────────

class JobType(str, Enum):
    optimize = "optimize"
    simulate = "simulate"
    whatif = "whatif"


class WorkloadRef(BaseModel):
    type: str = Field(..., description="'builtin' or 'trace'")
    name: Optional[str] = None        # for builtin: 'azure', 'lmsys', etc.
    trace_id: Optional[str] = None    # for trace uploads


class OptimizeParams(BaseModel):
    workload: WorkloadRef
    lam: float = Field(..., gt=0, description="Arrival rate (req/s)")
    slo_ms: float = Field(500.0, gt=0, description="P99 TTFT SLO (ms)")
    b_short: int = Field(4096, ge=1, description="Short/long context split (tokens)")
    gpu_short: str = Field("a100", description="GPU for short pool")
    gpu_long: str = Field("a100", description="GPU for long pool")
    long_max_ctx: int = Field(65536, ge=1024)
    gamma_min: float = Field(1.0, ge=1.0)
    gamma_max: float = Field(2.0, le=4.0)
    gamma_step: float = Field(0.1, ge=0.05)
    n_sim_requests: int = Field(20000, ge=1000)
    p_c: float = Field(
        0.75,
        ge=0.0, le=1.0,
        description=(
            "Effective compression success probability for the C&R analytical model. "
            "The greedy compressor has p_c=1.0 per safe request; multiply by the "
            "safe-category fraction of borderline traffic (e.g. 0.75 for a workload "
            "with 75%% prose/RAG/mixed and 25%% code).  Default 0.75."
        ),
    )
    node_avail: float = Field(
        1.0,
        gt=0.0, le=1.0,
        description=(
            "Steady-state fraction of GPU nodes that are healthy (availability). "
            "The optimizer inflates the raw SLO-sized GPU count by 1/node_avail "
            "so the SLO still holds while (1-node_avail) nodes are under repair. "
            "Compute from empirical failure data with: "
            "A = 1 / (1 + r_f_per_node_day * mttr_hours / 24). "
            "Meta RSC-1 reference (Kokolis et al. 2024, arXiv:2410.21680): "
            "r_f=0.0065/node-day; MTTR=4h → A≈0.9989, MTTR=48h → A≈0.9871. "
            "Default 1.0 = no reliability margin."
        ),
    )


class SimulateParams(BaseModel):
    workload: WorkloadRef
    fleet_id: Optional[str] = None   # use saved fleet
    fleet: Optional[FleetConfigIn] = None  # or inline fleet
    lam: float = Field(..., gt=0)
    slo_ms: float = Field(500.0, gt=0)
    n_requests: int = Field(20000, ge=100)


class WhatifParams(BaseModel):
    workload: WorkloadRef
    fleet_id: Optional[str] = None
    fleet: Optional[FleetConfigIn] = None
    lam_range: List[float] = Field(..., min_length=2)
    slo_ms: float = Field(500.0, gt=0)
    n_requests: int = Field(10000, ge=100)


class JobRequest(BaseModel):
    type: JobType
    optimize: Optional[OptimizeParams] = None
    simulate: Optional[SimulateParams] = None
    whatif: Optional[WhatifParams] = None


class JobStatus(str, Enum):
    pending = "pending"
    running = "running"
    done = "done"
    failed = "failed"


# ── Simulation result sub-models ──────────────────────────────────────────────

class PoolResult(BaseModel):
    pool_id: str
    gpu: str
    n_gpus: int
    p50_ttft_ms: float
    p99_ttft_ms: float
    p99_queue_wait_ms: float
    slo_compliance: float
    mean_utilisation: float
    cost_per_hr: float


class SimResult(BaseModel):
    total_gpus: int
    annual_cost_kusd: float
    fleet_p99_ttft_ms: float
    fleet_p50_ttft_ms: float
    fleet_slo_compliance: float
    fleet_mean_utilisation: float
    pools: List[PoolResult]
    ttft_histogram: List[HistogramBucket] = Field(default_factory=list)
    arrival_rate_actual: float


class SweepPoint(BaseModel):
    gamma: float
    n_s: int
    n_l: int
    total_gpus: int
    annual_cost_kusd: float
    p99_short_ms: float
    p99_long_ms: float
    slo_met: bool
    source: str


class OptResult(BaseModel):
    best: SweepPoint
    sweep: List[SweepPoint]
    baseline_annual_cost_kusd: float
    savings_pct: float
    sim_validation: Optional[SimResult] = None


class WhatifPoint(BaseModel):
    lam: float
    fleet_p99_ttft_ms: float
    fleet_slo_compliance: float
    fleet_mean_utilisation: float
    annual_cost_kusd: float


class WhatifResult(BaseModel):
    points: List[WhatifPoint]
    slo_break_lam: Optional[float] = None


class JobOut(BaseModel):
    id: str
    type: JobType
    status: JobStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    request: JobRequest
    result_optimize: Optional[OptResult] = None
    result_simulate: Optional[SimResult] = None
    result_whatif: Optional[WhatifResult] = None
