import type {
  CallRecord,
  Comparison,
  Catalog,
  CaseResult,
  EvidencePage,
  Dataset,
  Manifest,
  Plan,
  Report,
  Run,
  RunEvent,
  Target,
} from './types'

export const SR_BENCH_API = '/api/sr-bench/v1'

export class SrBenchRequestError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message)
    this.name = 'SrBenchRequestError'
  }
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const response = await fetch(`${SR_BENCH_API}${path}`, {
    ...init,
    headers: { 'Content-Type': 'application/json', ...init.headers },
  })
  const value: unknown = await response.json().catch(() => null)
  if (!response.ok) {
    const payload = value && typeof value === 'object' ? (value as Record<string, unknown>) : {}
    const detail =
      typeof payload.error === 'object' && payload.error
        ? (payload.error as Record<string, unknown>).message
        : payload.error
    throw new SrBenchRequestError(
      typeof detail === 'string'
        ? detail
        : typeof payload.message === 'string'
          ? payload.message
          : `sr-bench request failed (HTTP ${response.status}).`,
      response.status,
    )
  }
  if (value === null)
    throw new SrBenchRequestError('sr-bench returned an empty response.', response.status)
  return value as T
}

const post = <T>(path: string, body: unknown) =>
  request<T>(path, { method: 'POST', body: JSON.stringify(body) })
const runPath = (id: string) => `/runs/${encodeURIComponent(id)}`

export const benchApi = {
  catalog: (signal?: AbortSignal) => request<Catalog>('/catalog', { signal }),
  datasets: (signal?: AbortSignal) => request<{ datasets: Dataset[] }>('/datasets', { signal }),
  targets: (signal?: AbortSignal) => request<{ targets: Target[] }>('/targets', { signal }),
  runs: (signal?: AbortSignal) => request<{ runs: Run[] }>('/runs', { signal }),
  run: (id: string, signal?: AbortSignal) => request<Run>(runPath(id), { signal }),
  plan: (manifest: Manifest) => post<Plan>('/plans', { manifest }),
  start: (manifest: Manifest, idempotencyKey: string) =>
    post<Run>('/runs', { manifest, idempotency_key: idempotencyKey }),
  cancel: (id: string) => post<Run>(`${runPath(id)}/cancel`, {}),
  calls: (id: string, after = 0, signal?: AbortSignal) =>
    request<EvidencePage & { calls: CallRecord[] }>(
      `${runPath(id)}/calls?after=${after}&limit=100`,
      { signal },
    ),
  call: (id: string, callId: string, signal?: AbortSignal) =>
    request<CallRecord>(`${runPath(id)}/calls/${encodeURIComponent(callId)}`, { signal }),
  results: (id: string, after = 0, signal?: AbortSignal) =>
    request<EvidencePage & { results: CaseResult[] }>(
      `${runPath(id)}/results?after=${after}&limit=100`,
      { signal },
    ),
  report: (id: string, signal?: AbortSignal) =>
    request<Report>(`${runPath(id)}/report`, { signal }),
  events: (id: string, after = 0, signal?: AbortSignal) =>
    request<{ events: RunEvent[] }>(`${runPath(id)}/events?after=${after}`, { signal }),
  replay: (baseline: string, preview: string) =>
    post<Run>('/replays', { baseline_run_id: baseline, preview_run_id: preview }),
  regrade: (id: string) => post<Record<string, unknown>>(`${runPath(id)}/regrade`, {}),
  exportMatrix: (id: string) => post<Record<string, unknown>>(`${runPath(id)}/export`, {}),
  compare: (baseline: string, candidate: string) =>
    post<Comparison>('/comparisons', {
      baseline_run_id: baseline,
      candidate_run_id: candidate,
    }),
}
