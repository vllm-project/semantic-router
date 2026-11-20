import { describe, expect, it } from 'vitest'
import type { ExperimentMember } from './experimentApi'
import { experimentQuestionCount, experimentRoleLabels } from './experimentEvidencePresentation'
import {
  filterRuns,
  parseRunFilters,
  runBestScore,
  runCost,
  runFilterEntries,
  sequenceExperiment,
} from './runPresentation'
import type { ExperimentRunContext, Run, TargetMetrics } from './types'

function run(overrides: Partial<Run> = {}): Run {
  return {
    id: 'run-fixture',
    status: 'failed',
    created_at: '2026-01-01T00:00:00Z',
    updated_at: '2026-01-01T00:01:00Z',
    manifest: {
      version: 'sr-bench-1.0',
      name: 'Three-model reference',
      mode: 'live',
      profile: 'quick',
      seed: 7,
      targets: ['one', 'two', 'three'].map((id) => ({
        id,
        kind: 'single',
        base_url: 'https://example.com/v1',
        model: `${id}-model`,
      })),
      cases: Array.from({ length: 14 }, (_, index) => ({ id: `case-${index}` })),
      sampling: { temperature: 1, max_tokens: 100 },
      limits: {
        concurrency: 1,
        total_timeout_s: 30,
        idle_timeout_s: 10,
        max_output_tokens: 100,
        max_output_chars: 1000,
        repetition_window: 100,
        repetition_limit: 5,
        max_cost_usd: 1,
        max_run_seconds: 100,
        max_calls_per_case: 1,
      },
    },
    progress: { total: 42, completed: 41, failed: 1 },
    ...overrides,
  }
}

function member(
  role: ExperimentRunContext['role'],
  runID = 'run-fixture',
  linked = '2026-01-01T00:00:00Z',
): ExperimentMember {
  return { role, run_id: runID, hypothesis: '', linked_at: linked }
}

const complete = (id: string, accuracy: number, cost?: number): TargetMetrics => ({
  id,
  total: 14,
  completed: 14,
  scored: 14,
  failed: 0,
  pending: 0,
  macro_accuracy: accuracy,
  accuracy,
  ...(cost === undefined ? {} : { cost_usd: cost }),
})

describe('Experiment sequence', () => {
  it('orders main roles as a workflow regardless of link order and numbers versions', () => {
    const roles = ['validation', 'candidate', 'baseline', 'candidate', 'initial'] as const
    const members = roles.map((role, index) =>
      member(role, `${role}-${index}`, `2026-01-0${5 - index}T00:00:00Z`),
    )
    const runs = members.map((item) => run({ id: item.run_id, created_at: item.linked_at }))
    expect(sequenceExperiment(members, runs).map((item) => item.label)).toEqual([
      'Reference',
      'Starting recipe',
      'Version 1',
      'Version 2',
      'Final validation',
    ])
  })

  it('keeps supporting checks after the main sequence under their own labels', () => {
    const roles = ['recovery', 'baseline', 'preview', 'smoke', 'estimate'] as const
    const labels = sequenceExperiment(
      roles.map((role) => member(role, role)),
      roles.map((role) => run({ id: role })),
    ).map((item) => item.label)
    expect(labels[0]).toBe('Reference')
    expect(labels.slice(1).sort()).toEqual(
      ['recovery', 'preview', 'smoke', 'estimate']
        .map((role) => experimentRoleLabels[role as ExperimentRunContext['role']])
        .sort(),
    )
  })

  it('labels a candidate without claiming that it improved the result', () => {
    const [item] = sequenceExperiment([member('candidate')], [run()])
    expect(experimentRoleLabels.candidate).toBe('Recipe version')
    expect(item.label).toBe('Version 1')
    expect(item.label).not.toMatch(/improved|better|winner/i)
  })

  it('keeps a saved member in the sequence when its run summary is unavailable', () => {
    const [item] = sequenceExperiment([member('candidate', 'unloaded-private-id')], [])
    expect(item.run).toBeUndefined()
    expect(item.member.run_id).toBe('unloaded-private-id')
  })
})

describe('Run table values', () => {
  it('reports the best score only from targets whose every planned outcome is scored', () => {
    const value = run({
      summary: {
        targets: [complete('one', 0.5), { ...complete('two', 0.9), pending: 2, scored: 12 }],
      },
    })
    expect(runBestScore(value)).toEqual({ score: 50, target: 'one-model' })
    expect(runBestScore(run({ summary: { targets: [] } }))).toBeNull()
  })

  it('never scores a dry run or an offline estimate', () => {
    const preview = run({ summary: { targets: [complete('one', 0.8)] } })
    preview.manifest.mode = 'preview'
    expect(runBestScore(preview)).toBeNull()
  })

  it('keeps cost unknown when any target has no recorded cost', () => {
    expect(
      runCost(
        run({ summary: { targets: [complete('one', 0.5, 0.25), complete('two', 0.4, 0.5)] } }),
      ),
    ).toBe(0.75)
    expect(
      runCost(run({ summary: { targets: [complete('one', 0.5, 0.25), complete('two', 0.4)] } })),
    ).toBeNull()
    expect(runCost(run())).toBeNull()
  })

  it('confirms questions from the frozen dataset rather than attempt totals', () => {
    const value = run()
    delete value.manifest.cases
    value.manifest.dataset = {
      path: 'frozen.jsonl',
      sha256: 'fixture',
      case_count: 14,
    } as NonNullable<Run['manifest']['dataset']>
    expect(experimentQuestionCount(value)).toBe(14)
  })

  it('does not derive questions from attempt totals, recovery subsets or conflicting evidence', () => {
    const value = run()
    delete value.manifest.cases
    expect(experimentQuestionCount(value)).toBeUndefined()
    value.manifest.cases = [{ id: 'one' }]
    value.manifest.dataset = {
      path: 'frozen.jsonl',
      sha256: 'fixture',
      case_count: 14,
    } as NonNullable<Run['manifest']['dataset']>
    expect(experimentQuestionCount(value)).toBeUndefined()
    delete value.manifest.dataset
    value.manifest.recovery = { recovery_subset: true }
    expect(experimentQuestionCount(value)).toBeUndefined()
    delete value.manifest.recovery
    value.manifest.execution_cells = [{ case_id: 'one', target_id: 'one' }]
    expect(experimentQuestionCount(value)).toBeUndefined()
  })

  it.each([0, -1, 1.5, '14', Number.NaN, Number.POSITIVE_INFINITY])(
    'does not turn invalid metadata %s into a displayed count',
    (caseCount) => {
      const value = run()
      delete value.manifest.cases
      value.manifest.dataset = {
        path: 'frozen.jsonl',
        sha256: 'fixture',
        case_count: caseCount,
      } as NonNullable<Run['manifest']['dataset']>
      expect(experimentQuestionCount(value)).toBeUndefined()
    },
  )
})

describe('Run filters in the URL', () => {
  it('reads comma lists, drops unknown values and keeps the older active alias', () => {
    const filters = parseRunFilters(
      new URLSearchParams('status=active,failed,bogus&mode=live,other&profile=quick&q=Ref&page=2'),
    )
    expect(filters).toEqual({
      query: 'Ref',
      status: ['running', 'queued', 'failed'],
      mode: ['live'],
      benchmark: [],
      profile: ['quick'],
      page: 2,
    })
  })

  it('writes empty filters as removed parameters', () => {
    expect(
      runFilterEntries({ status: [], query: '', page: 0, profile: ['smoke', 'quick'] }),
    ).toEqual([
      ['status', null],
      ['q', null],
      ['page', null],
      ['profile', 'smoke,quick'],
    ])
  })

  it('matches benchmarks from the frozen dataset and searches names, ids and models', () => {
    const value = run({ status: 'completed' })
    value.manifest.dataset = {
      path: 'p',
      sha256: 's',
      benchmarks: ['mmlu-pro'],
    } as NonNullable<Run['manifest']['dataset']>
    const base = parseRunFilters(new URLSearchParams())
    expect(filterRuns([value], { ...base, benchmark: ['mmlu-pro'] })).toHaveLength(1)
    expect(filterRuns([value], { ...base, benchmark: ['gpqa-diamond'] })).toHaveLength(0)
    expect(filterRuns([value], { ...base, query: 'two-model' })).toHaveLength(1)
    expect(filterRuns([value], { ...base, status: ['failed'] })).toHaveLength(0)
  })
})
