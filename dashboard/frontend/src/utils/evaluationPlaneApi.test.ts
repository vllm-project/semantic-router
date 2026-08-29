import { afterEach, describe, expect, it, vi } from 'vitest'

import type {
  CreateEvaluationRunRequest,
  EvaluationCatalog,
  EvaluationRun,
  EvaluationRunEvent,
} from '../types/evaluationPlane'
import {
  buildCreateRunPayload,
  cancelEvaluationRun,
  compareEvaluationRuns,
  createEvaluationRun,
  deleteEvaluationRun,
  getEvaluationCatalog,
  getEvaluationArtifactURL,
  getEvaluationReport,
  getEvaluationRun,
  isDownloadableEvaluationArtifact,
  listEvaluationRuns,
  startEvaluationRun,
  subscribeToEvaluationRun,
} from './evaluationPlaneApi'

const catalog: EvaluationCatalog = {
  schema_version: 'evaluation.v1',
  gate_contract_version: 'evaluation-release-gates.v1',
  change_profiles: [
    {
      id: 'recipe',
      name: 'Routing recipe',
      description: 'Recipe signal, decision, algorithm, and policy changes.',
    },
  ],
  tracks: [
    {
      id: 'routing',
      name: 'Routing',
      description: 'Routing quality',
      modes: ['replay'],
      metrics: [],
    },
  ],
  suites: [
    {
      id: 'suite-routing',
      name: 'Routing suite',
      description: 'Replay suite',
      track_ids: ['routing'],
      modes: ['replay'],
      evidence_level: 'E2',
    },
  ],
  targets: [
    {
      id: 'target-approved',
      name: 'Approved target',
      description: 'Server target',
      kind: 'replay',
      track_ids: ['routing'],
      modes: ['replay'],
    },
  ],
}

const request: CreateEvaluationRunRequest = {
  name: ' Candidate ',
  description: ' Compare recipe ',
  suite_ids: ['suite-routing'],
  track_ids: ['routing'],
  mode: 'replay',
  target_id: 'target-approved',
  change_profile: 'recipe',
  sample_limit: 25,
  concurrency: 2,
  seed: 42,
  auto_start: true,
}

const run: EvaluationRun = {
  schema_version: 'evaluation.v1',
  id: 'run-1',
  name: 'Candidate',
  description: 'Compare recipe',
  status: 'pending',
  mode: 'replay',
  evidence_level: 'E2',
  target_id: 'target-approved',
  change_profile: 'recipe',
  suite_ids: ['suite-routing'],
  track_ids: ['routing'],
  sample_limit: 25,
  concurrency: 2,
  seed: 42,
  progress: { percent: 0, completed: 0, total: 1 },
  created_at: '2026-08-29T00:00:00Z',
}

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json' },
  })
}

afterEach(() => vi.unstubAllGlobals())

describe('Evaluation Plane API', () => {
  it('keeps native SSE reconnect active, deduplicates event ids, and stops at terminal events', () => {
    class FakeEventSource {
      static readonly CONNECTING = 0
      static readonly OPEN = 1
      static readonly CLOSED = 2
      static instances: FakeEventSource[] = []

      readonly listeners = new Map<string, EventListener[]>()
      readonly close = vi.fn(() => {
        this.readyState = FakeEventSource.CLOSED
      })
      readyState = FakeEventSource.CONNECTING
      onmessage: ((event: MessageEvent<string>) => void) | null = null
      onerror: ((event: Event) => void) | null = null

      constructor(readonly url: string) {
        FakeEventSource.instances.push(this)
      }

      addEventListener(name: string, listener: EventListener) {
        this.listeners.set(name, [...(this.listeners.get(name) || []), listener])
      }

      emit(name: string, event: EvaluationRunEvent) {
        const message = { data: JSON.stringify(event) } as MessageEvent<string>
        this.listeners.get(name)?.forEach((listener) => listener(message))
      }

      fail(readyState: number) {
        this.readyState = readyState
        this.onerror?.({ type: 'error' } as Event)
      }
    }

    vi.stubGlobal('EventSource', FakeEventSource)
    const onEvent = vi.fn()
    const onTerminal = vi.fn()
    const onError = vi.fn()
    const unsubscribe = subscribeToEvaluationRun('run 1', onEvent, onTerminal, onError)
    const source = FakeEventSource.instances[0]

    expect(source?.url).toBe('/api/evaluation/v1/runs/run%201/events')
    source?.fail(FakeEventSource.CONNECTING)
    expect(source?.close).not.toHaveBeenCalled()
    expect(onError).not.toHaveBeenCalled()

    const progress: EvaluationRunEvent = {
      id: 'event-1',
      run_id: 'run-1',
      type: 'progress',
      timestamp: '2026-08-29T00:00:00Z',
      message: 'Routing track started',
    }
    source?.emit('progress', progress)
    source?.emit('progress', progress)
    expect(onEvent).toHaveBeenCalledTimes(1)

    source?.emit('completed', { ...progress, id: 'event-2', type: 'completed' })
    expect(onEvent).toHaveBeenCalledTimes(2)
    expect(onTerminal).toHaveBeenCalledTimes(1)
    expect(source?.close).toHaveBeenCalledTimes(1)
    source?.emit('progress', { ...progress, id: 'event-3' })
    expect(onEvent).toHaveBeenCalledTimes(2)

    unsubscribe()
  })

  it('terminates a server-closed SSE stream instead of retrying it', () => {
    class ClosedEventSource {
      static readonly CONNECTING = 0
      static readonly OPEN = 1
      static readonly CLOSED = 2
      static instance: ClosedEventSource | null = null

      readonly close = vi.fn()
      readyState = ClosedEventSource.CONNECTING
      onmessage: ((event: MessageEvent<string>) => void) | null = null
      onerror: ((event: Event) => void) | null = null

      constructor(readonly url: string) {
        ClosedEventSource.instance = this
      }

      addEventListener() {}
    }

    vi.stubGlobal('EventSource', ClosedEventSource)
    const onError = vi.fn()
    subscribeToEvaluationRun('run-1', vi.fn(), vi.fn(), onError)
    const source = ClosedEventSource.instance
    if (!source) throw new Error('Expected the EventSource test double to be constructed.')

    source.readyState = ClosedEventSource.CLOSED
    source.onerror?.({ type: 'error' } as Event)

    expect(source.close).toHaveBeenCalledTimes(1)
    expect(onError).toHaveBeenCalledWith(
      new Error('Evaluation event stream was closed by the server.'),
    )
  })

  it('only serializes catalog-backed target ids and public request fields', () => {
    const untrusted = {
      ...request,
      endpoint: 'https://arbitrary.invalid/v1',
      url: 'https://arbitrary.invalid',
      hidden_label: 'must never cross the browser contract',
    } as CreateEvaluationRunRequest
    const payload = buildCreateRunPayload(untrusted, catalog)

    expect(payload).toEqual({
      ...request,
      name: 'Candidate',
      description: 'Compare recipe',
      auto_start: false,
    })
    expect(payload).not.toHaveProperty('endpoint')
    expect(payload).not.toHaveProperty('url')
    expect(payload).not.toHaveProperty('hidden_label')
    expect(() =>
      buildCreateRunPayload({ ...request, target_id: 'https://arbitrary.invalid' }, catalog),
    ).toThrow(/server evaluation catalog/i)
    expect(() =>
      buildCreateRunPayload({ ...request, change_profile: 'selector' }, catalog),
    ).toThrow(/change profile.*server evaluation catalog/i)
  })

  it('rejects partially supported suites and tracks before creating a run', () => {
    const expandedCatalog: EvaluationCatalog = {
      ...catalog,
      tracks: [
        ...catalog.tracks,
        {
          id: 'agentic',
          name: 'Agentic',
          description: 'Trajectory evidence',
          modes: ['replay'],
          metrics: [],
        },
      ],
      suites: [
        ...catalog.suites,
        {
          id: 'suite-partial',
          name: 'Partially supported suite',
          description: 'Requires routing and agentic evidence',
          track_ids: ['routing', 'agentic'],
          modes: ['replay'],
          evidence_level: 'E2',
        },
      ],
    }

    expect(() =>
      buildCreateRunPayload(
        {
          ...request,
          suite_ids: ['suite-partial'],
          track_ids: ['routing'],
        },
        expandedCatalog,
      ),
    ).toThrow(/fully supported/i)
    expect(() =>
      buildCreateRunPayload(
        {
          ...request,
          track_ids: ['agentic'],
        },
        expandedCatalog,
      ),
    ).toThrow(/selected track/i)
  })

  it('links only backend-allowlisted report artifacts and never the run manifest', () => {
    for (const name of [
      'routing-traces.jsonl',
      'capacity-profile.json',
      'metrics.json',
      'gates.json',
      'comparison.json',
      'failure-summary.json',
      'provenance.json',
      'checksums.sha256',
    ]) {
      expect(isDownloadableEvaluationArtifact({ id: name, name, kind: 'evidence' })).toBe(true)
    }
    expect(
      isDownloadableEvaluationArtifact({
        id: 'records.jsonl',
        name: 'Case records',
        kind: 'records',
      }),
    ).toBe(false)
    expect(
      isDownloadableEvaluationArtifact({
        id: 'failure-summary-json',
        name: 'failure-summary.json',
        kind: 'json',
      }),
    ).toBe(true)
    expect(
      isDownloadableEvaluationArtifact({
        id: 'report-html',
        name: 'Rendered report',
        kind: 'html',
      }),
    ).toBe(false)
    expect(
      isDownloadableEvaluationArtifact({
        id: 'run-manifest.json',
        name: 'Run manifest',
        kind: 'manifest',
      }),
    ).toBe(false)
    expect(getEvaluationArtifactURL('run one', 'records/visible.jsonl')).toBe(
      '/api/evaluation/v1/runs/run%20one/artifacts/records%2Fvisible.jsonl',
    )
  })

  it('uses only the versioned catalog and run lifecycle endpoints', async () => {
    const fetchMock = vi
      .fn<typeof fetch>()
      .mockResolvedValueOnce(jsonResponse(catalog))
      .mockResolvedValueOnce(jsonResponse({ runs: [run] }))
      .mockResolvedValueOnce(jsonResponse({ run }))
      .mockResolvedValueOnce(jsonResponse({ run }, 201))
      .mockResolvedValueOnce(jsonResponse({ run: { ...run, status: 'running' } }))
      .mockResolvedValueOnce(jsonResponse({ run: { ...run, status: 'cancelled' } }))
      .mockResolvedValueOnce(new Response(null, { status: 204 }))
      .mockResolvedValueOnce(jsonResponse({ run, summary: {} }))
      .mockResolvedValueOnce(
        jsonResponse({ baseline_run_id: 'base', candidate_run_id: 'candidate' }),
      )
    vi.stubGlobal('fetch', fetchMock)

    await getEvaluationCatalog()
    await listEvaluationRuns()
    await getEvaluationRun('run 1')
    await createEvaluationRun(request, catalog)
    await startEvaluationRun('run 1')
    await cancelEvaluationRun('run 1')
    await deleteEvaluationRun('run 1')
    await getEvaluationReport('run 1')
    await compareEvaluationRuns('base run', 'candidate/run')

    expect(fetchMock.mock.calls.map(([url]) => url)).toEqual([
      '/api/evaluation/v1/catalog',
      '/api/evaluation/v1/runs',
      '/api/evaluation/v1/runs/run%201',
      '/api/evaluation/v1/runs',
      '/api/evaluation/v1/runs/run%201/start',
      '/api/evaluation/v1/runs/run%201/cancel',
      '/api/evaluation/v1/runs/run%201',
      '/api/evaluation/v1/runs/run%201/report',
      '/api/evaluation/v1/compare?baseline_run_id=base+run&candidate_run_id=candidate%2Frun',
    ])
    expect(fetchMock.mock.calls[3]?.[1]).toMatchObject({ method: 'POST' })
    expect(JSON.parse(String(fetchMock.mock.calls[3]?.[1]?.body))).toMatchObject({
      auto_start: false,
      change_profile: 'recipe',
    })
    expect(fetchMock.mock.calls[5]?.[1]).toMatchObject({ method: 'POST' })
    expect(fetchMock.mock.calls[6]?.[1]).toMatchObject({ method: 'DELETE' })
  })
})
