import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'
import ExperimentEvidence from './ExperimentEvidence'
import type { ExperimentMember } from './experimentApi'
import { experimentQuestionCount, experimentRoleLabels } from './experimentEvidencePresentation'
import type { ExperimentRunContext, Run } from './types'

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
        model: id,
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

function member(role: ExperimentRunContext['role'], runID = 'run-fixture'): ExperimentMember {
  return { role, run_id: runID, hypothesis: '', linked_at: '2026-01-01T00:00:00Z' }
}

function render(members: ExperimentMember[], runs: Run[], hasMore = false): string {
  return renderToStaticMarkup(
    <ExperimentEvidence members={members} runs={runs} hasMore={hasMore} onOpenRun={() => {}} />,
  )
}

describe('Experiment saved evidence', () => {
  it('distinguishes questions from attempts without inventing quality or costs', () => {
    const html = render([member('baseline')], [run()])
    expect(html).toContain('41 / 42')
    expect(html).toContain('attempts completed')
    expect(html).toContain('1 failed')
    expect(html).toContain('<dt>Questions</dt><dd>14</dd>')
    expect(html).toContain('<dt>Mode</dt><dd>Live</dd>')
    expect(html).toContain('<dt>Profile</dt><dd>Quick</dd>')
    expect(html).toContain('>failed</span>')
    expect(html).not.toContain('run-fixture')
    expect(html).not.toMatch(/accuracy|saving|\$/i)
  })

  it('groups main roles in workflow order regardless of link order', () => {
    const roles = ['validation', 'candidate', 'baseline', 'initial'] as const
    const html = render(
      roles.map((role) => member(role, role)),
      roles.map((role) => run({ id: role })),
    )
    const headings = [...html.matchAll(/<h4[^>]*>([^<]+)<\/h4>/g)].map((match) => match[1])
    expect(headings).toEqual([
      'Single-model reference',
      'Starting recipe',
      'Recipe versions',
      'Final validation',
    ])
    expect(html).not.toContain('Supporting checks')
    expect(html.match(/Single-model reference/g)).toHaveLength(1)
    expect(html).not.toMatch(/<h3[^>]*>Saved runs<\/h3>/)
  })

  it('keeps every supporting role in a collapsed, keyboard-native disclosure', () => {
    const roles = ['preview', 'smoke', 'estimate', 'recovery'] as const
    const html = render(
      roles.map((role) => member(role, role)),
      roles.map((role) => run({ id: role })),
    )
    expect(html).toContain('Supporting checks')
    expect(html.match(/<details\b[^>]*>/)?.[0]).not.toContain('open')
    const details = html.slice(html.indexOf('<details'), html.indexOf('</details>'))
    for (const role of roles) expect(details).toContain(experimentRoleLabels[role])
    expect(html).toContain('4 on this page')
  })

  it('labels a candidate without claiming that it improved the result', () => {
    const html = render([member('candidate')], [run()])
    expect(experimentRoleLabels.candidate).toBe('Recipe version')
    expect(html).toContain('Recipe version')
    expect(html).not.toMatch(/improved|better|winner/i)
  })

  it('presents the note as a change being tested and escapes saved content', () => {
    const html = render(
      [{ ...member('candidate'), hypothesis: 'Use <cheaper> models\nKeep the same cases.' }],
      [run()],
    )
    expect(html).toContain('Change being tested')
    expect(html).toContain('Use &lt;cheaper&gt; models')
    expect(html).not.toContain('Hypothesis')
    expect(render([{ ...member('candidate'), hypothesis: '   ' }], [run()])).not.toContain(
      'Change being tested',
    )
  })

  it('shows page scope and does not claim an empty experiment from an empty page', () => {
    expect(render([member('candidate')], [run()], true)).toContain(
      'More saved runs are available on the next page.',
    )
    const emptyPage = render([], [], true)
    expect(emptyPage).toContain('No saved runs on this page yet.')
    expect(emptyPage).toContain('Continue to the next page')
    expect(emptyPage).not.toContain('Start with a single-model reference')
    expect(render([], [])).toContain('or add a run you have already saved')
  })

  it('keeps a saved member visible when its run summary is unavailable', () => {
    const html = render([member('candidate', 'unloaded-private-id')], [])
    expect(html).toContain('Saved run')
    expect(html).toContain('Open this run to view its status and progress.')
    expect(html).not.toContain('unloaded-private-id')
    expect(html).not.toContain('attempts completed')
    expect(html).not.toContain('0 failed')
  })
})

describe('Confirmed question count', () => {
  it('uses frozen dataset metadata when collection responses omit raw cases', () => {
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
