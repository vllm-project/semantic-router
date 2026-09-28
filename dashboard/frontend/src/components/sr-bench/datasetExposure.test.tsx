import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it, vi } from 'vitest'
import DatasetDetails from './DatasetDetails'
import DatasetExposure from './DatasetExposure'
import DatasetInventory from './DatasetInventory'
import RunDetails from './RunDetails'
import { useRunEvidence } from './useRunEvidence'
import type { Dataset, DatasetDetail, DatasetPreparation, Report, Run } from './types'

vi.mock('./useRunEvidence', () => ({ useRunEvidence: vi.fn() }))

const mixed: DatasetPreparation = {
  scicode: {
    evaluation_role: 'holdout',
    coverage: 'named-memberships-only',
    selected_count: 8,
    excluded_count: 2,
  },
  'gpqa-diamond': {
    evaluation_role: 'retest',
    coverage: 'no-history-qualification',
    selected_count: 6,
    excluded_count: 0,
  },
}

function dataset(preparation?: DatasetPreparation): Dataset {
  return {
    id: 'synthetic-dataset',
    name: 'Synthetic suite',
    path: 'synthetic.jsonl',
    sha256: 'private-case-hash',
    case_count: 14,
    profile: 'standard',
    split: 'holdout',
    benchmarks: Object.keys(preparation ?? mixed),
    preparation,
  }
}

function inventory(value: Dataset): string {
  return renderToStaticMarkup(
    <DatasetInventory datasets={[value]} runs={[]} canRun={false} onUse={() => {}} />,
  )
}

function runDetails(preparation?: DatasetPreparation, reportPreparation?: DatasetPreparation) {
  const run: Run = {
    id: 'synthetic-run',
    status: 'completed',
    created_at: '2026-01-01T00:00:00Z',
    updated_at: '2026-01-01T00:01:00Z',
    progress: { total: 0, completed: 0, failed: 0 },
    manifest: {
      version: 'sr-bench-1.0',
      name: 'Synthetic run',
      mode: 'live',
      profile: 'standard',
      seed: 7,
      targets: [],
      dataset: dataset(preparation),
      limits: {
        concurrency: 1,
        total_timeout_s: 30,
        idle_timeout_s: 10,
        max_output_chars: 1000,
        repetition_window: 100,
        repetition_limit: 5,
        max_cost_usd: 1,
        max_run_seconds: 100,
        max_calls_per_case: 1,
      },
      sampling: { temperature: 1 },
    },
  }
  const report: Report = {
    version: 'sr-bench-1.0',
    run_id: run.id,
    status: 'completed',
    summary: { targets: [] },
    benchmarks: [],
    limitations: ['General scoring limitation.', 'Later unstructured limitation.'],
    provenance: { dataset: dataset(reportPreparation) },
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
  const html = renderToStaticMarkup(
    <RunDetails
      id={run.id}
      actorID="synthetic-viewer"
      canRun={false}
      onSectionChange={() => {}}
      onChanged={() => {}}
      onRecovered={() => {}}
      onCandidate={() => {}}
    />,
  )
  return html.slice(
    html.indexOf('id="run-panel-results"'),
    html.indexOf('id="run-panel-questions"'),
  )
}

describe('Dataset evaluation role presentation', () => {
  it('labels a mixed suite with concise family counts instead of calling the whole split Holdout', () => {
    const html = inventory(dataset(mixed))
    expect(html).toContain('Mixed evaluation roles')
    expect(html).toContain('1 Holdout · 1 Retest')
    expect(html).not.toContain('<strong>GPQA Diamond</strong>: Retest')
    expect(html).not.toContain('>Holdout</span>')
    expect(html).not.toContain('private-case-hash')
  })

  it('keeps nine-family cards compact without duplicating benchmark names', () => {
    const preparation = Object.fromEntries(
      Array.from({ length: 9 }, (_, index) => [
        `family-${index}`,
        index < 7 ? mixed.scicode : mixed['gpqa-diamond'],
      ]),
    )
    const html = renderToStaticMarkup(<DatasetExposure preparation={preparation} compact />)
    expect(html).toContain('7 Holdout · 2 Retest')
    expect(html).not.toMatch(/family-|<li|selected|excluded/)
  })

  it('labels an explicitly declared retest even when its frozen split is holdout', () => {
    const html = inventory(dataset({ 'gpqa-diamond': mixed['gpqa-diamond'] }))
    expect(html).toContain('>Retest</span>')
    expect(html).not.toContain('Holdout')
  })

  it('does not infer an evaluation role from split when the structured role is unspecified', () => {
    const preparation = { scicode: { ...mixed.scicode, evaluation_role: null } }
    const html = inventory(dataset(preparation))
    expect(html).toContain('Role unspecified')
    expect(html).not.toContain('Holdout')
  })

  it('preserves ordinary datasets without inventing exposure qualification', () => {
    const html = inventory(dataset())
    expect(html).toContain('>Standard</span>')
    expect(html).not.toMatch(/Holdout|Retest|Role unspecified|Mixed evaluation roles/)
    expect(renderToStaticMarkup(<DatasetExposure />)).toBe('')
    expect(renderToStaticMarkup(<DatasetExposure preparation={{}} />)).toBe('')
  })

  it('keeps missing family provenance distinct instead of extending another family’s holdout role', () => {
    const value = { ...dataset({ scicode: mixed.scicode }), benchmarks: ['scicode', 'hle'] }
    const html = inventory(value)
    expect(html).toContain('Mixed evaluation roles')
    expect(html).toContain('1 Holdout · 1 Unspecified')
    expect(html).not.toContain('>Holdout</span>')
    const summary = renderToStaticMarkup(
      <DatasetExposure preparation={value.preparation} benchmarks={value.benchmarks} />,
    )
    expect(summary).toContain('Role unspecified · No preparation provenance')
  })

  it('shows known collection provenance on dataset details while the detail read is loading', () => {
    const value = dataset(mixed)
    const html = renderToStaticMarkup(
      <DatasetDetails
        id={value.id}
        dataset={value}
        canRun={false}
        onUse={() => {}}
        onBack={() => {}}
      />,
    )
    expect(html).toContain('Mixed evaluation roles')
    expect(html).toContain('aria-label="Dataset evaluation roles"')
    expect(html).toContain('Combined results are not an unseen holdout aggregate.')
    expect(html).not.toContain('Holdout set')
  })

  it('presents detail provenance as family roles, counts and scope without dumping proofs', () => {
    const proof = {
      ...mixed.scicode,
      history_snapshot: { id: 'private-history-hash', task_keys: ['private-task-id'] },
      task_source: { revision: 'private-source-hash' },
    }
    const provenance: DatasetDetail['provenance'] = {
      sources: [],
      preparation: { ...mixed, scicode: proof },
    }
    const html = renderToStaticMarkup(<DatasetExposure preparation={provenance.preparation} />)
    expect(html).toContain('8 selected · 2 excluded · Recorded history checked')
    expect(html).toContain('6 selected · 0 excluded · No history qualification')
    expect(html).toContain('History exclusions cover only the frozen dataset and run memberships.')
    expect(html).toContain('Other prior exposure is not ruled out.')
    expect(html).not.toMatch(/private-|history_snapshot|task_source|<pre|<details/)
  })

  it('does not call an all-holdout dataset an unseen or contamination-free guarantee', () => {
    const html = renderToStaticMarkup(<DatasetExposure preparation={{ scicode: mixed.scicode }} />)
    expect(html).toContain('<strong>SciCode</strong>: Holdout')
    expect(html).toContain('Other prior exposure is not ruled out.')
    expect(html).not.toContain('Includes retest families')
  })

  it.each(['manifest', 'report'] as const)(
    'shows retest disclosure in visible results from structured %s provenance independently of limitations',
    (source) => {
      const html = runDetails(
        source === 'manifest' ? mixed : undefined,
        source === 'report' ? mixed : undefined,
      )
      expect(html).toContain('<strong>GPQA Diamond</strong>: Retest')
      expect(html).toContain('<strong>SciCode</strong>: Holdout')
      expect(html).toContain('Combined results are not an unseen holdout aggregate.')
      expect(html).toContain('General scoring limitation.')
      expect(html).not.toMatch(
        /Later unstructured limitation|More limitations|<pre|private-case-hash/,
      )
    },
  )

  it('adds no exposure summary to an ordinary saved run without provenance', () => {
    expect(runDetails()).not.toContain('Dataset evaluation roles')
  })
})
