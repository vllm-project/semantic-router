import type {
  CallRecord,
  Comparison,
  Catalog,
  CaseResult,
  EvidencePage,
  Dataset,
  DatasetDetail,
  DatasetCasePage,
  Manifest,
  Plan,
  Report,
  RecoveryPlan,
  RecoveryRequest,
  Run,
  RunEvent,
  Target,
} from './types'

export const SR_BENCH_API = '/api/sr-bench/v1'

export class SrBenchRequestError extends Error {
  constructor(
    message: string,
    readonly status: number,
    readonly code?: string,
    readonly dispatchStarted?: boolean,
  ) {
    super(message)
    this.name = 'SrBenchRequestError'
  }
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  // Bound reads and initial run submission responses, including stalled bodies.
  // A submission timeout is ambiguous; its durable identity must be reconciled.
  // No request is automatically retried by this layer.
  const reading = !init.method || ['GET', 'HEAD'].includes(init.method)
  const submitting = path === '/runs' && init.method === 'POST'
  const preparing = ['/plans', '/datasets/compose'].includes(path) && init.method === 'POST'
  const computing =
    (path === '/comparisons' || path.endsWith('/recover-plan')) && init.method === 'POST'
  const controller = reading || submitting || preparing || computing ? new AbortController() : null
  let timedOut = false
  const abort = () => controller?.abort()
  init.signal?.addEventListener('abort', abort, { once: true })
  if (init.signal?.aborted) abort()
  const deadline = controller
    ? setTimeout(() => {
        timedOut = true
        controller.abort()
      }, 30000)
    : undefined
  try {
    const response = await fetch(`${SR_BENCH_API}${path}`, {
      ...init,
      signal: controller?.signal ?? init.signal,
      headers: { 'Content-Type': 'application/json', ...init.headers },
    })
    const value: unknown = await response.json().catch(() => null)
    if (timedOut) throw new Error('Read deadline exceeded')
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
        typeof payload.code === 'string' ? payload.code : undefined,
        typeof payload.dispatch_started === 'boolean' ? payload.dispatch_started : undefined,
      )
    }
    if (value === null)
      throw new SrBenchRequestError('sr-bench returned an empty response.', response.status)
    return value as T
  } catch (error) {
    if (timedOut)
      throw new SrBenchRequestError(
        submitting
          ? 'The evaluation submission response timed out. Reconcile the saved submission before starting another attempt.'
          : preparing
            ? 'Preparing the evaluation timed out after 30 seconds. No model generation was requested. You can review again using the same selection.'
            : 'Reading saved sr-bench evidence timed out after 30 seconds. Existing runs continue independently.',
        408,
      )
    throw error
  } finally {
    if (deadline) clearTimeout(deadline)
    init.signal?.removeEventListener('abort', abort)
  }
}

const post = <T>(path: string, body: unknown) =>
  request<T>(path, { method: 'POST', body: JSON.stringify(body) })
const runPath = (id: string) => `/runs/${encodeURIComponent(id)}`

export const benchApi = {
  catalog: (signal?: AbortSignal) => request<Catalog>('/catalog', { signal }),
  datasets: (signal?: AbortSignal) => request<{ datasets: Dataset[] }>('/datasets', { signal }),
  dataset: (id: string, signal?: AbortSignal) =>
    request<DatasetDetail>(`/datasets/${encodeURIComponent(id)}`, { signal }),
  datasetCases: (
    id: string,
    filters: {
      cursor?: string
      limit?: number
      benchmark?: string
      category?: string
      q?: string
    } = {},
    signal?: AbortSignal,
  ) => {
    const query = new URLSearchParams()
    for (const [key, value] of Object.entries(filters)) {
      if (value !== undefined && value !== '') query.set(key, String(value))
    }
    return request<DatasetCasePage>(`/datasets/${encodeURIComponent(id)}/cases?${query}`, {
      signal,
    })
  },
  composeDatasets: (datasetIDs: string[], benchmarks: string[]) =>
    post<{ dataset: Dataset }>('/datasets/compose', { dataset_ids: datasetIDs, benchmarks }),
  targets: (signal?: AbortSignal) => request<{ targets: Target[] }>('/targets', { signal }),
  runs: (signal?: AbortSignal) => request<{ runs: Run[] }>('/runs', { signal }),
  run: (id: string, signal?: AbortSignal) => request<Run>(runPath(id), { signal }),
  plan: (manifest: Manifest) => post<Plan>('/plans', { manifest }),
  start: (manifest: Manifest, idempotencyKey: string) =>
    post<Run>('/runs', { manifest, idempotency_key: idempotencyKey }),
  recoveryPlan: (id: string, mode: RecoveryPlan['mode']) =>
    post<RecoveryPlan>(`${runPath(id)}/recover-plan`, { mode }),
  recover: (id: string, recovery: RecoveryRequest) => post<Run>(`${runPath(id)}/recover`, recovery),
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
