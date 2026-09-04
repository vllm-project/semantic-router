import type {
  CreateEvaluationRunRequest,
  EvaluationCatalog,
  EvaluationComparison,
  EvaluationReport,
  EvaluationRun,
  EvaluationRunEvent,
} from '../types/evaluationPlane'

export const EVALUATION_API_BASE = '/api/evaluation/v1'

type RunEnvelope = { run: EvaluationRun }
type RunsEnvelope = { runs: EvaluationRun[] }

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === 'object' && !Array.isArray(value)
}

async function readError(response: Response): Promise<string> {
  const fallback = `Evaluation request failed (HTTP ${response.status})`
  try {
    const payload: unknown = await response.json()
    if (!isRecord(payload)) return fallback
    if (typeof payload.message === 'string') return payload.message
    if (typeof payload.error === 'string') return payload.error
    if (isRecord(payload.error) && typeof payload.error.message === 'string') {
      return payload.error.message
    }
  } catch {
    // Fall through to the status-based message.
  }
  return fallback
}

async function requestJson<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${EVALUATION_API_BASE}${path}`, init)
  if (!response.ok) throw new Error(await readError(response))
  if (response.status === 204) return undefined as T
  return response.json() as Promise<T>
}

function unwrapRun(payload: EvaluationRun | RunEnvelope): EvaluationRun {
  return isRecord(payload) && 'run' in payload ? (payload as RunEnvelope).run : payload
}

export function getEvaluationCatalog(signal?: AbortSignal): Promise<EvaluationCatalog> {
  return requestJson('/catalog', { signal })
}

export async function listEvaluationRuns(signal?: AbortSignal): Promise<EvaluationRun[]> {
  const payload = await requestJson<EvaluationRun[] | RunsEnvelope>('/runs', { signal })
  return Array.isArray(payload) ? payload : payload.runs
}

export async function getEvaluationRun(id: string, signal?: AbortSignal): Promise<EvaluationRun> {
  const payload = await requestJson<EvaluationRun | RunEnvelope>(
    `/runs/${encodeURIComponent(id)}`,
    { signal },
  )
  return unwrapRun(payload)
}

export function buildCreateRunPayload(
  request: CreateEvaluationRunRequest,
  catalog: EvaluationCatalog,
): CreateEvaluationRunRequest {
  const changeProfile = catalog.change_profiles.find(
    (candidate) => candidate.id === request.change_profile,
  )
  if (!changeProfile) {
    throw new Error('Select a change profile from the server evaluation catalog.')
  }

  const target = catalog.targets.find((candidate) => candidate.id === request.target_id)
  if (!target) {
    throw new Error('Select a target from the server evaluation catalog.')
  }

  if (!target.modes.includes(request.mode) || target.healthy === false) {
    throw new Error('The selected target cannot execute this evaluation mode.')
  }

  const suitesByID = new Map(catalog.suites.map((suite) => [suite.id, suite]))
  const suites = request.suite_ids.map((suiteID) => suitesByID.get(suiteID))
  if (suites.some((suite) => !suite)) {
    throw new Error('One or more selected suites are no longer in the evaluation catalog.')
  }

  if (
    suites.some(
      (suite) =>
        !suite?.modes.includes(request.mode) ||
        suite.track_ids.some((trackID) => !target.track_ids.includes(trackID)),
    )
  ) {
    throw new Error('Every selected suite must be fully supported by the target and mode.')
  }

  const suiteTrackIDs = new Set(suites.flatMap((suite) => suite?.track_ids || []))
  if (
    request.track_ids.some(
      (trackID) => !target.track_ids.includes(trackID) || !suiteTrackIDs.has(trackID),
    )
  ) {
    throw new Error('Every selected track must be supported by the target and selected suites.')
  }

  return {
    name: request.name.trim(),
    description: request.description.trim(),
    suite_ids: [...new Set(request.suite_ids)],
    track_ids: [...new Set(request.track_ids)],
    mode: request.mode,
    target_id: request.target_id,
    change_profile: changeProfile.id,
    sample_limit: Math.max(1, Math.floor(request.sample_limit)),
    concurrency: Math.max(1, Math.floor(request.concurrency)),
    seed: Math.floor(request.seed),
    ...(request.baseline_run_id ? { baseline_run_id: request.baseline_run_id } : {}),
    // Creating a run is evaluation.write; execution is separately protected by
    // evaluation.run. Never let a create payload bypass the /start boundary.
    auto_start: false,
  }
}

export async function createEvaluationRun(
  request: CreateEvaluationRunRequest,
  catalog: EvaluationCatalog,
): Promise<EvaluationRun> {
  const payload = await requestJson<EvaluationRun | RunEnvelope>('/runs', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(buildCreateRunPayload(request, catalog)),
  })
  return unwrapRun(payload)
}

export async function startEvaluationRun(id: string): Promise<EvaluationRun> {
  const payload = await requestJson<EvaluationRun | RunEnvelope>(
    `/runs/${encodeURIComponent(id)}/start`,
    { method: 'POST' },
  )
  return unwrapRun(payload)
}

export async function cancelEvaluationRun(id: string): Promise<EvaluationRun> {
  const payload = await requestJson<EvaluationRun | RunEnvelope>(
    `/runs/${encodeURIComponent(id)}/cancel`,
    { method: 'POST' },
  )
  return unwrapRun(payload)
}

export function deleteEvaluationRun(id: string): Promise<void> {
  return requestJson(`/runs/${encodeURIComponent(id)}`, { method: 'DELETE' })
}

export function getEvaluationReport(id: string, signal?: AbortSignal): Promise<EvaluationReport> {
  return requestJson(`/runs/${encodeURIComponent(id)}/report`, { signal })
}

const DOWNLOADABLE_ARTIFACT_NAMES = new Set([
  'routing-traces.jsonl',
  'capacity-profile.json',
  'metrics.json',
  'gates.json',
  'comparison.json',
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
  return `${EVALUATION_API_BASE}/runs/${encodeURIComponent(runID)}/artifacts/${encodeURIComponent(artifactID)}`
}

export function compareEvaluationRuns(
  baselineRunID: string,
  candidateRunID: string,
  signal?: AbortSignal,
): Promise<EvaluationComparison> {
  const query = new URLSearchParams({
    baseline_run_id: baselineRunID,
    candidate_run_id: candidateRunID,
  })
  return requestJson(`/compare?${query.toString()}`, { signal })
}

export function subscribeToEvaluationRun(
  id: string,
  onEvent: (event: EvaluationRunEvent) => void,
  onTerminal: () => void,
  onError: (error: Error) => void,
): () => void {
  const source = new EventSource(`${EVALUATION_API_BASE}/runs/${encodeURIComponent(id)}/events`)
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
      const event = JSON.parse(message.data) as EvaluationRunEvent
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
