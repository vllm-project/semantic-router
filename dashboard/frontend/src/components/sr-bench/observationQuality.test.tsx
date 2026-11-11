import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it, vi } from 'vitest'
import RunDetails from './RunDetails'
import { DEFAULT_LIMITS, hasScoredOutcomes, qualityApplies } from './model'
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

const sessionObservation: TargetMetrics = {
  ...observation,
  result_kind: 'session-observation-v1',
  session_observation: {
    planned_sessions: 1,
    attempted_sessions: 1,
    completed_sessions: 1,
    failed_sessions: 0,
    planned_turns: 2,
    attempted_turns: 2,
    completed_turns: 2,
    failed_turns: 0,
    undispatched_turns: 0,
    execution_complete: true,
    success_rate: 1,
    latency_ms: { p50: 12, p95: 14, p95_assessable: true },
    selected_model_counts: { 'model-a': 2 },
    unknown_selected_model_turns: 0,
    selected_model_transitions: 1,
    selected_model_switches: 0,
    selected_model_continuity_rate: 1,
    tool_loop_switch_violations: 0,
    planned_tool_loop_transitions: 0,
    assessed_tool_loop_transitions: 0,
    usage_complete: true,
    usage: { input_tokens: 4, output_tokens: 2, cached_input_tokens: 0 },
    cache_evidence_qualification_counts: { unknown: 2 },
    cache_evidence_unknown_turns: 2,
    phase_metrics: {
      user_turn: {
        planned_turns: 2,
        attempted_turns: 2,
        http_successes: 2,
        http_outcomes_unknown: 0,
        latency_ms: { p50: 12, p95: 14 },
        request_phase_counts: { user_turn: 2 },
        router_phase_counts: { 'phase-a': 2 },
      },
    },
    session_recovery: {
      error_recovery_status: 'not_measured',
      error_recovery_reason: 'CallFailure ends the session case.',
      idle_resumption: {
        planned: 0,
        attempted: 0,
        successful: 0,
        request_session_id_reused: 0,
      },
    },
    validation: {
      status: 'pass',
      predicates: {
        min_success_rate: {
          enabled: false,
          required: 0,
          actual: 1,
          status: 'pass',
          reason: 'threshold_disabled',
        },
      },
    },
  },
}

function renderMetrics(
  metrics: TargetMetrics,
  status: Run['status'] = 'completed',
  targets: TargetMetrics[] = [metrics],
): string {
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
    summary: { targets },
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

describe('non-binary quality presentation', () => {
  it('shows missing cache evidence separately from the unknown counter', () => {
    const html = renderMetrics({
      ...sessionObservation,
      session_observation: {
        ...sessionObservation.session_observation!,
        cache_evidence_qualification_counts: { missing: 36 },
        cache_evidence_unknown_turns: 0,
      },
    })
    expect(html).toContain('Cache evidence qualifications:')
    expect(html).toContain('missing: 36')
    expect(html).toContain('Cache evidence unknown for 0 planned turns.')
  })

  it('labels mixed rows with scored or observed counts over their planned denominators', () => {
    const binary: TargetMetrics = {
      ...observation,
      id: 'binary',
      result_kind: undefined,
      total: 3,
      scored: 2,
      observed: undefined,
      correct: 1,
    }
    const session: TargetMetrics = { ...sessionObservation, id: 'session', total: 4, observed: 1 }
    const html = renderMetrics(
      { ...observation, result_kind: 'mixed', total: 7, scored: 2, observed: 1 },
      'completed',
      [binary, session],
    )
    expect(html).toContain('<th>Scored or observed / planned</th>')
    expect(html).not.toContain('<th>Scored / observed</th>')
    expect(html).toContain('<td>2 / 3</td>')
    expect(html).toContain('<td>1 / 4</td>')
  })

  it.each([
    observation,
    sessionObservation,
    { ...observation, result_kind: 'mixed', total: 2, completed: 2, scored: 1 },
  ])('does not label a completed $result_kind report as incomplete quality', (metrics) => {
    expect(hasScoredOutcomes(metrics)).toBe(false)
    const html = renderMetrics(metrics)
    expect(html).toContain('>Not applicable<')
    expect(html).not.toContain('>Incomplete<')
    expect(html).not.toContain('95% CI')
    expect(html).not.toContain('failures count as incorrect')
    expect(metrics.scored).toBe(metrics.result_kind === 'mixed' ? 1 : 0)
    if (metrics.result_kind === 'session-observation-v1') {
      expect(html).toContain('Session-routing observations')
      expect(html).toContain('Session recovery after a failed request was not measured.')
      expect(html).toContain('Workload and Router response phases')
      expect(html).toContain('12 / 14 ms')
    }
  })

  it('keeps quality inapplicable when the observation sequence itself fails', () => {
    const html = renderMetrics(
      { ...observation, completed: 0, observed: 0, failed: 1, complete: false },
      'failed',
    )
    expect(html).toContain('>Not applicable<')
    expect(html).toContain('>failed<')
  })

  it('does not present MRCR sequence ratios as binary accuracy', () => {
    const mrcr: TargetMetrics = {
      ...observation,
      result_kind: 'mrcr-sequence-ratio-v1',
      total: 3,
      completed: 3,
      failed: 0,
      pending: 0,
      scored: 3,
      observed: undefined,
      correct: null,
      accuracy: null,
      accuracy_ci95: null,
      complete: true,
    }
    expect(qualityApplies(mrcr)).toBe(false)
    expect(hasScoredOutcomes(mrcr)).toBe(true)
    const html = renderMetrics(mrcr)
    expect(html).toContain('>Not applicable<')
    expect(html).toContain('<td>3 / 3</td>')
    expect(html).not.toContain('<td>— / 3</td>')
    expect(html).not.toContain('95% CI')
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
