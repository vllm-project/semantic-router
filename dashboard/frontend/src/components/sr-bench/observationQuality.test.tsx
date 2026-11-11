import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it, vi } from 'vitest'
import RunDetails from './RunDetails'
import { DEFAULT_LIMITS, hasScoredOutcomes } from './model'
import type { Report, Run, TargetMetrics } from './types'
import { useRunEvidence } from './useRunEvidence'

vi.mock('./useRunEvidence', () => ({ useRunEvidence: vi.fn() }))

const observation: TargetMetrics = {
  id: 'subject',
  result_kind: 'cache-observation-v1',
  total: 1,
  completed: 1,
  failed: 0,
  pending: 0,
  scored: 0,
  observed: 1,
  complete: true,
  correct: null,
  accuracy: null,
  accuracy_ci95: null,
  macro_accuracy: null,
  sr_bench_score: null,
}

function renderMetrics(metrics: TargetMetrics, status: Run['status'] = 'completed'): string {
  const run: Run = {
    id: 'observation-run',
    status,
    created_at: '2026-01-01T00:00:00Z',
    updated_at: '2026-01-01T00:01:00Z',
    progress: {
      total: metrics.total ?? 0,
      completed: metrics.completed ?? 0,
      failed: metrics.failed ?? 0,
    },
    manifest: {
      version: 'sr-bench-1.0',
      name: 'Cache observation',
      mode: 'live',
      profile: 'quick',
      seed: 0,
      targets: [
        {
          id: 'subject',
          kind: 'single',
          model: 'requested-model',
          base_url: 'http://127.0.0.1:8000/v1',
        },
      ],
      limits: DEFAULT_LIMITS,
      sampling: { temperature: 0, max_tokens: 16 },
    },
  }
  const report: Report = {
    version: 'sr-bench-1.0',
    run_id: run.id,
    status,
    summary: { targets: [metrics] },
    benchmarks: [{ ...metrics, benchmark: 'cache-token-probe-v1' }],
    limitations: [],
    provenance: {},
  }
  const page = { total: 0, nextCursor: null, loading: false, error: '' }
  vi.mocked(useRunEvidence).mockReturnValue({
    run,
    report,
    reportRead: { loading: false, error: '' },
    readAt: null,
    results: [],
    events: [],
    calls: [],
    resultsPage: page,
    eventsPage: page,
    callsPage: page,
    loadMoreResults: async () => {},
    loadMoreEvents: async () => {},
    loadMoreCalls: async () => {},
    error: '',
  })
  return renderToStaticMarkup(
    <RunDetails
      id={run.id}
      actorID="viewer"
      canRun={false}
      onSectionChange={() => {}}
      onChanged={() => {}}
      onRecovered={() => {}}
      onCandidate={() => {}}
    />,
  )
}

describe('observation-only quality presentation', () => {
  it.each([
    observation,
    { ...observation, result_kind: 'mixed', total: 2, completed: 2, scored: 1 },
  ])('does not label a completed $result_kind report as incomplete quality', (metrics) => {
    expect(hasScoredOutcomes(metrics)).toBe(false)
    const html = renderMetrics(metrics)
    expect(html).toContain('>Not applicable<')
    expect(html).not.toContain('>Incomplete<')
    expect(html).not.toContain('95% CI')
    expect(html).not.toContain('failures count as incorrect')
    expect(metrics.scored).toBe(metrics.result_kind === 'mixed' ? 1 : 0)
  })

  it('keeps quality inapplicable when the observation sequence itself fails', () => {
    const html = renderMetrics(
      { ...observation, completed: 0, observed: 0, failed: 1, complete: false },
      'failed',
    )
    expect(html).toContain('>Not applicable<')
    expect(html).toContain('>failed<')
  })

  it('keeps incomplete and pending quality states for ordinary ungraded runs', () => {
    const binary = {
      ...observation,
      result_kind: undefined,
      observed: undefined,
      correct: 0,
      complete: false,
    }
    expect(renderMetrics(binary)).toContain('>Incomplete<')
    expect(renderMetrics(binary, 'running')).toContain('>Pending<')
    expect(renderMetrics(binary)).not.toContain('>Not applicable<')
  })
})
