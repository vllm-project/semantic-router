export const FLEET_SIM_PROXY_PREFIX = '/api/fleet-sim'
export const FLEET_SIM_API_PREFIX = `${FLEET_SIM_PROXY_PREFIX}/api`

export const FLEET_SIM_NAV_ITEMS = [
  { label: 'Overview', to: '/fleet-sim' },
  { label: 'Workloads', to: '/fleet-sim/workloads' },
  { label: 'Fleets', to: '/fleet-sim/fleets' },
  { label: 'Runs', to: '/fleet-sim/runs' },
] as const

export type FleetSimTraceFormat = 'semantic_router' | 'jsonl' | 'csv'
export type FleetSimJobType = 'optimize' | 'simulate' | 'whatif'
export type FleetSimJobStatus = 'pending' | 'running' | 'done' | 'failed'

export interface HistogramBucket {
  lo: number
  hi: number
  count: number
}

export interface TraceStats {
  n_requests: number
  duration_s: number
  arrival_rate_rps: number
  p50_prompt_tokens: number
  p95_prompt_tokens: number
  p99_prompt_tokens: number
  p50_output_tokens: number
  p99_output_tokens: number
  p50_total_tokens: number
  p99_total_tokens: number
  routing_distribution: Record<string, number>
  prompt_histogram: HistogramBucket[]
  output_histogram: HistogramBucket[]
}

export interface BuiltinWorkload {
  name: string
  description: string
  path: string
  stats?: TraceStats | null
}

export interface TraceInfo {
  id: string
  name: string
  format: FleetSimTraceFormat
  upload_time: string
  n_requests: number
  stats?: TraceStats | null
}

export interface TraceSample {
  records: Array<Record<string, unknown>>
  total: number
}

export interface GpuProfile {
  name: string
  W_ms: number
  H_ms_per_slot: number
  chunk: number
  blk_size: number
  total_kv_blks: number
  max_slots: number
  cost_per_hr: number
}

export interface PoolConfig {
  pool_id: string
  gpu: string
  n_gpus: number
  max_ctx: number
}

export interface FleetConfig {
  id: string
  name: string
  pools: PoolConfig[]
  router: string
  compress_gamma?: number | null
  created_at: string
  total_gpus: number
  estimated_cost_per_hr: number
  estimated_annual_cost_kusd: number
}

export interface WorkloadRef {
  type: 'builtin' | 'trace'
  name?: string
  trace_id?: string
}

export interface SweepPoint {
  gamma: number
  n_s: number
  n_l: number
  total_gpus: number
  annual_cost_kusd: number
  p99_short_ms: number
  p99_long_ms: number
  slo_met: boolean
  source: string
}

export interface PoolResult {
  pool_id: string
  gpu: string
  n_gpus: number
  p50_ttft_ms: number
  p99_ttft_ms: number
  p99_queue_wait_ms: number
  slo_compliance: number
  mean_utilisation: number
  cost_per_hr: number
}

export interface SimResult {
  total_gpus: number
  annual_cost_kusd: number
  fleet_p99_ttft_ms: number
  fleet_p50_ttft_ms: number
  fleet_slo_compliance: number
  fleet_mean_utilisation: number
  pools: PoolResult[]
  ttft_histogram: HistogramBucket[]
  arrival_rate_actual: number
}

export interface WhatifPoint {
  lam: number
  fleet_p99_ttft_ms: number
  fleet_slo_compliance: number
  fleet_mean_utilisation: number
  annual_cost_kusd: number
}

export interface WhatifResult {
  points: WhatifPoint[]
  slo_break_lam?: number | null
}

export interface FleetSimJob {
  id: string
  type: FleetSimJobType
  status: FleetSimJobStatus
  created_at: string
  started_at?: string | null
  completed_at?: string | null
  error?: string | null
  request: {
    type: FleetSimJobType
    optimize?: Record<string, unknown>
    simulate?: Record<string, unknown>
    whatif?: Record<string, unknown>
  }
  result_optimize?: {
    best: SweepPoint
    sweep: SweepPoint[]
    baseline_annual_cost_kusd: number
    savings_pct: number
    sim_validation?: SimResult | null
  } | null
  result_simulate?: SimResult | null
  result_whatif?: WhatifResult | null
}

async function fleetSimRequest<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${FLEET_SIM_PROXY_PREFIX}${path}`, init)
  if (!response.ok) {
    let message = `Fleet simulator request failed (${response.status})`
    const text = await response.text()
    if (text.trim()) {
      try {
        const payload = JSON.parse(text)
        if (typeof payload?.detail === 'string') message = payload.detail
        else if (typeof payload?.message === 'string') message = payload.message
        else if (typeof payload?.error === 'string') message = payload.error
        else message = text.trim()
      } catch {
        message = text.trim()
      }
    }
    throw new Error(message)
  }
  if (response.status === 204) {
    return undefined as T
  }
  return response.json() as Promise<T>
}

export function listWorkloads(): Promise<BuiltinWorkload[]> {
  return fleetSimRequest('/api/workloads')
}

export function getWorkloadStats(name: string): Promise<TraceStats> {
  return fleetSimRequest(`/api/workloads/${encodeURIComponent(name)}/stats`)
}

export function listTraces(): Promise<TraceInfo[]> {
  return fleetSimRequest('/api/traces')
}

export function getTraceSample(traceID: string, limit = 20): Promise<TraceSample> {
  return fleetSimRequest(`/api/traces/${encodeURIComponent(traceID)}/sample?limit=${limit}`)
}

export async function uploadTrace(file: File, format: FleetSimTraceFormat): Promise<TraceInfo> {
  const body = new FormData()
  body.append('file', file)
  return fleetSimRequest(`/api/traces?fmt=${format}`, {
    method: 'POST',
    body,
  })
}

export async function deleteTrace(traceID: string): Promise<void> {
  await fleetSimRequest(`/api/traces/${encodeURIComponent(traceID)}`, {
    method: 'DELETE',
  })
}

export function listGpuProfiles(): Promise<GpuProfile[]> {
  return fleetSimRequest('/api/gpu-profiles')
}

export function listFleets(): Promise<FleetConfig[]> {
  return fleetSimRequest('/api/fleets')
}

export function createFleet(body: {
  name: string
  pools: PoolConfig[]
  router: string
  compress_gamma?: number | null
}): Promise<FleetConfig> {
  return fleetSimRequest('/api/fleets', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
}

export async function deleteFleet(fleetID: string): Promise<void> {
  await fleetSimRequest(`/api/fleets/${encodeURIComponent(fleetID)}`, {
    method: 'DELETE',
  })
}

export function listJobs(): Promise<FleetSimJob[]> {
  return fleetSimRequest('/api/jobs')
}

export function getJob(jobID: string): Promise<FleetSimJob> {
  return fleetSimRequest(`/api/jobs/${encodeURIComponent(jobID)}`)
}

export function createJob(body: Record<string, unknown>): Promise<FleetSimJob> {
  return fleetSimRequest('/api/jobs', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
}

export async function deleteJob(jobID: string): Promise<void> {
  await fleetSimRequest(`/api/jobs/${encodeURIComponent(jobID)}`, {
    method: 'DELETE',
  })
}
