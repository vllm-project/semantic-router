"""Fleet configuration management routes."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from .. import storage
from ..models import FleetConfigIn, FleetConfigOut, GpuProfileOut

router = APIRouter(tags=["Fleets"])

_GPU_PROFILES = {
    "a100": {
        "name": "A100-80GB",
        "W_ms": 4.0,
        "H_ms_per_slot": 0.32,
        "chunk": 512,
        "blk_size": 16,
        "total_kv_blks": 4000,
        "max_slots": 256,
        "cost_per_hr": 2.21,
    },
    "h100": {
        "name": "H100-80GB",
        "W_ms": 2.8,
        "H_ms_per_slot": 0.22,
        "chunk": 512,
        "blk_size": 16,
        "total_kv_blks": 4000,
        "max_slots": 256,
        "cost_per_hr": 3.89,
    },
    "a10g": {
        "name": "A10G-24GB",
        "W_ms": 12.0,
        "H_ms_per_slot": 0.90,
        "chunk": 512,
        "blk_size": 16,
        "total_kv_blks": 1440,
        "max_slots": 128,
        "cost_per_hr": 1.01,
    },
}


def _cost_per_hr(pools: list) -> float:
    total = 0.0
    for p in pools:
        gpu = _GPU_PROFILES.get(p["gpu"].lower(), {})
        total += gpu.get("cost_per_hr", 0) * p["n_gpus"]
    return total


@router.get(
    "/gpu-profiles",
    response_model=list[GpuProfileOut],
    tags=["Fleets"],
    summary="List available GPU profiles",
)
async def list_gpu_profiles():
    return [GpuProfileOut(**v) for v in _GPU_PROFILES.values()]


@router.get("/fleets", response_model=list[FleetConfigOut], summary="List saved fleets")
async def list_fleets():
    return [FleetConfigOut(**f) for f in storage.list_fleets()]


@router.post(
    "/fleets", response_model=FleetConfigOut, summary="Save a fleet configuration"
)
async def create_fleet(body: FleetConfigIn):
    fleet_id = storage.new_id()
    pools_data = [p.model_dump() for p in body.pools]
    cost_hr = _cost_per_hr(pools_data)
    data = {
        "id": fleet_id,
        "name": body.name,
        "pools": pools_data,
        "router": body.router,
        "compress_gamma": body.compress_gamma,
        "created_at": storage.now_iso(),
        "total_gpus": sum(p.n_gpus for p in body.pools),
        "estimated_cost_per_hr": round(cost_hr, 4),
        "estimated_annual_cost_kusd": round(cost_hr * 8760 / 1000, 1),
    }
    storage.save_fleet(fleet_id, data)
    return FleetConfigOut(**data)


@router.get("/fleets/{fleet_id}", response_model=FleetConfigOut, summary="Get a fleet")
async def get_fleet(fleet_id: str):
    data = storage.get_fleet(fleet_id)
    if not data:
        raise HTTPException(404, "Fleet not found")
    return FleetConfigOut(**data)


@router.delete("/fleets/{fleet_id}", summary="Delete a fleet")
async def delete_fleet(fleet_id: str):
    if not storage.delete_fleet(fleet_id):
        raise HTTPException(404, "Fleet not found")
    return {"deleted": fleet_id}
