import type {
  CreateEvaluationRunPayload,
  EvaluationCatalog,
  EvaluationCatalogChangeProfile,
  EvaluationRun,
  EvaluationRunEvent,
  EvaluationRunLedger,
} from '../types/evaluationPlane'
import type {
  CreateEvaluationCampaignPayload,
  EvaluationCampaign,
  EvaluationCampaignReadiness,
} from '../types/evaluationCampaign'
import type { EvaluationComparison } from '../types/evaluationComparison'
import type { EvaluationReport } from '../types/evaluationReport'
import type {
  CreateEvaluationControlledPairPayload,
  EvaluationControlledPairExecution,
} from '../types/evaluationControlledPair'
import {
  buildCreateEvaluationCampaignPayload,
  decodeEvaluationCampaign,
} from './evaluationCampaignContract'
import {
  decodeEvaluationCampaignReadiness,
  mergeEvaluationCampaignReadinessPages,
  type EvaluationCampaignReadinessAnchors,
} from './evaluationCampaignReadinessContract'
import { decodeEvaluationControlledPairExecution } from './evaluationControlledPairContract'
import { decodeEvaluationCatalog } from './evaluationCatalogContract'
import { buildCreateRunPayload } from './evaluationCreateRunContract'
import { decodeEvaluationComparison } from './evaluationComparisonContract'
import { decodeEvaluationReport } from './evaluationReportContract'
import {
  decodeEvaluationRun,
  decodeEvaluationRunEvent,
  decodeEvaluationRunLedger,
  requireCanonicalEvaluationRunID,
} from './evaluationRunContract'

const EVALUATION_API_BASE = '/api/evaluation/v1'

export class EvaluationRequestError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message)
    this.name = 'EvaluationRequestError'
  }
}

function record(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === 'object' && !Array.isArray(value)
}

async function readError(response: Response): Promise<string> {
  const fallback = `Evaluation request failed (HTTP ${response.status})`
  try {
    const payload: unknown = await response.json()
    if (!record(payload)) return fallback
    if (typeof payload.message === 'string') return payload.message
    if (typeof payload.error === 'string') return payload.error
    if (record(payload.error) && typeof payload.error.message === 'string') {
      return payload.error.message
    }
  } catch {
    // The HTTP status remains the authoritative fallback for non-JSON failures.
  }
  return fallback
}

async function requestJson<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${EVALUATION_API_BASE}${path}`, init)
  if (!response.ok) throw new EvaluationRequestError(await readError(response), response.status)
  if (response.status === 204) return undefined as T
  return response.json() as Promise<T>
}

export async function getEvaluationCatalog(signal?: AbortSignal): Promise<EvaluationCatalog> {
  return decodeEvaluationCatalog(await requestJson<unknown>('/catalog', { signal }))
}

interface ListEvaluationRunsOptions {
  cursor?: string
  signal?: AbortSignal
}

export async function listEvaluationRuns({
  cursor,
  signal,
}: ListEvaluationRunsOptions = {}): Promise<EvaluationRunLedger> {
  const query = cursor ? `?${new URLSearchParams({ cursor }).toString()}` : ''
  return decodeEvaluationRunLedger(await requestJson<unknown>(`/runs${query}`, { signal }), cursor)
}

export async function getEvaluationRun(id: string, signal?: AbortSignal): Promise<EvaluationRun> {
  requireCanonicalEvaluationRunID(id)
  return decodeEvaluationRun(
    await requestJson<unknown>(`/runs/${encodeURIComponent(id)}`, { signal }),
    id,
  )
}

export async function createEvaluationRun(
  request: CreateEvaluationRunPayload,
  catalog: EvaluationCatalog,
): Promise<EvaluationRun> {
  return decodeEvaluationRun(
    await requestJson<unknown>('/runs', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(buildCreateRunPayload(request, catalog)),
    }),
  )
}

export async function startEvaluationRun(id: string): Promise<EvaluationRun> {
  requireCanonicalEvaluationRunID(id)
  return decodeEvaluationRun(
    await requestJson<unknown>(`/runs/${encodeURIComponent(id)}/start`, { method: 'POST' }),
    id,
  )
}

export async function cancelEvaluationRun(id: string): Promise<EvaluationRun> {
  requireCanonicalEvaluationRunID(id)
  return decodeEvaluationRun(
    await requestJson<unknown>(`/runs/${encodeURIComponent(id)}/cancel`, { method: 'POST' }),
    id,
  )
}

export function deleteEvaluationRun(id: string): Promise<void> {
  requireCanonicalEvaluationRunID(id)
  return requestJson(`/runs/${encodeURIComponent(id)}`, { method: 'DELETE' })
}

export async function getEvaluationReport(
  id: string,
  signal?: AbortSignal,
): Promise<EvaluationReport> {
  requireCanonicalEvaluationRunID(id)
  return decodeEvaluationReport(
    await requestJson<unknown>(`/runs/${encodeURIComponent(id)}/report`, { signal }),
    id,
  )
}

const DOWNLOADABLE_ARTIFACT_NAMES = new Set([
  'capacity-profile.json',
  'metrics.json',
  'gates.json',
  'failure-summary.json',
  'provenance.json',
  'checksums.sha256',
])

export function isDownloadableEvaluationArtifact(artifact: {
  id: string
  name: string
  kind: string
}): boolean {
  return DOWNLOADABLE_ARTIFACT_NAMES.has(artifact.name.toLowerCase())
}

export function getEvaluationArtifactURL(runID: string, artifactID: string): string {
  requireCanonicalEvaluationRunID(runID)
  return `${EVALUATION_API_BASE}/runs/${encodeURIComponent(runID)}/artifacts/${encodeURIComponent(artifactID)}`
}

export async function getEvaluationArtifactJSON<T>(
  runID: string,
  artifactID: string,
  signal?: AbortSignal,
): Promise<T> {
  const response = await fetch(getEvaluationArtifactURL(runID, artifactID), { signal })
  if (!response.ok) throw new Error(await readError(response))
  const contentType = response.headers.get('Content-Type') || ''
  if (!contentType.toLowerCase().includes('application/json')) {
    throw new Error('Evaluation artifact did not return JSON evidence.')
  }
  return response.json() as Promise<T>
}

export async function compareEvaluationRuns(
  baselineRunID: string,
  candidateRunID: string,
  signal?: AbortSignal,
): Promise<EvaluationComparison> {
  requireCanonicalEvaluationRunID(baselineRunID)
  requireCanonicalEvaluationRunID(candidateRunID)
  const query = new URLSearchParams({
    baseline_run_id: baselineRunID,
    candidate_run_id: candidateRunID,
  })
  return decodeEvaluationComparison(
    await requestJson<unknown>(`/compare?${query.toString()}`, { signal }),
    baselineRunID,
    candidateRunID,
  )
}

export async function createEvaluationCampaign(
  request: CreateEvaluationCampaignPayload,
): Promise<EvaluationCampaign> {
  const payload = buildCreateEvaluationCampaignPayload(request)
  return decodeEvaluationCampaign(
    await requestJson<unknown>('/campaigns', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    }),
    payload.client_request_id,
  )
}

export async function getEvaluationCampaignReadiness(
  profile: EvaluationCatalogChangeProfile,
  anchors: EvaluationCampaignReadinessAnchors = {},
  signal?: AbortSignal,
): Promise<EvaluationCampaignReadiness> {
  if (anchors.controlledPairBaselineRunID) {
    requireCanonicalEvaluationRunID(anchors.controlledPairBaselineRunID)
  }
  if (anchors.fidelityReferenceRunID) {
    requireCanonicalEvaluationRunID(anchors.fidelityReferenceRunID)
  }
  const pages: EvaluationCampaignReadiness[] = []
  const seenCursors = new Set<string>()
  let cursor = ''
  do {
    const page = decodeEvaluationCampaignReadiness(
      await requestJson<unknown>('/campaign-readiness', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          change_profile: profile.id,
          limit: 200,
          ...(cursor ? { cursor } : {}),
          ...(anchors.controlledPairBaselineRunID
            ? { controlled_pair_baseline_run_id: anchors.controlledPairBaselineRunID }
            : {}),
          ...(anchors.fidelityReferenceRunID
            ? { fidelity_reference_run_id: anchors.fidelityReferenceRunID }
            : {}),
        }),
        signal,
      }),
      profile,
      anchors,
    )
    pages.push(page)
    cursor = page.next_cursor || ''
    if (cursor && seenCursors.has(cursor)) {
      throw new Error('Evaluation campaign readiness pagination did not advance.')
    }
    if (cursor) seenCursors.add(cursor)
  } while (cursor)
  return mergeEvaluationCampaignReadinessPages(pages)
}

export async function createEvaluationControlledPair(
  request: CreateEvaluationControlledPairPayload,
): Promise<EvaluationControlledPairExecution> {
  return decodeEvaluationControlledPairExecution(
    await requestJson<unknown>('/controlled-pairs', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
    }),
    request.client_request_id,
    request,
  )
}

export async function cancelEvaluationControlledPair(
  id: string,
): Promise<EvaluationControlledPairExecution> {
  requireCanonicalEvaluationRunID(id)
  return decodeEvaluationControlledPairExecution(
    await requestJson<unknown>(`/controlled-pairs/${encodeURIComponent(id)}/cancel`, {
      method: 'POST',
    }),
    id,
  )
}

export async function getEvaluationControlledPair(
  id: string,
  signal?: AbortSignal,
): Promise<EvaluationControlledPairExecution> {
  requireCanonicalEvaluationRunID(id)
  return decodeEvaluationControlledPairExecution(
    await requestJson<unknown>(`/controlled-pairs/${encodeURIComponent(id)}`, { signal }),
    id,
  )
}

export function deleteEvaluationControlledPair(id: string): Promise<void> {
  requireCanonicalEvaluationRunID(id)
  return requestJson(`/controlled-pairs/${encodeURIComponent(id)}`, { method: 'DELETE' })
}

export async function getEvaluationCampaign(
  id: string,
  signal?: AbortSignal,
): Promise<EvaluationCampaign> {
  requireCanonicalEvaluationRunID(id)
  return decodeEvaluationCampaign(
    await requestJson<unknown>(`/campaigns/${encodeURIComponent(id)}`, { signal }),
    id,
  )
}

export function subscribeToEvaluationRun(
  run: EvaluationRun,
  onEvent: (event: EvaluationRunEvent) => void,
  onTerminal: () => void,
  onError: (error: Error) => void,
): () => void {
  requireCanonicalEvaluationRunID(run.id)
  const source = new EventSource(`${EVALUATION_API_BASE}/runs/${encodeURIComponent(run.id)}/events`)
  let closed = false
  const seenEventIDs = new Set<string>()
  const eventNames = [
    'snapshot',
    'progress',
    'track',
    'gate',
    'artifact',
    'completed',
    'failed',
    'cancelled',
  ]

  const handleEvent = (message: MessageEvent<string>) => {
    if (closed) return
    try {
      const event = decodeEvaluationRunEvent(JSON.parse(message.data) as unknown, run)
      if (event.id && seenEventIDs.has(event.id)) return
      if (event.id) seenEventIDs.add(event.id)
      onEvent(event)
      if (['completed', 'failed', 'cancelled'].includes(event.type)) {
        closed = true
        source.close()
        onTerminal()
      }
    } catch {
      onError(new Error('Evaluation event stream returned an invalid event.'))
    }
  }
  eventNames.forEach((name) => source.addEventListener(name, handleEvent as EventListener))
  source.onmessage = handleEvent
  source.onerror = () => {
    if (closed || source.readyState !== EventSource.CLOSED) return
    closed = true
    source.close()
    onError(new Error('Evaluation event stream was closed by the server.'))
  }

  return () => {
    closed = true
    source.close()
  }
}
