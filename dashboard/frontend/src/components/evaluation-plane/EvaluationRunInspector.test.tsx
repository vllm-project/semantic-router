import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'

import type {
  EvaluationControlledPairExecution,
  EvaluationControlledPairState,
} from '../../types/evaluationControlledPair'
import type { EvaluationRun } from '../../types/evaluationPlane'
import EvaluationRunInspector from './EvaluationRunInspector'

const run: EvaluationRun = {
  schema_version: 'evaluation.v1',
  id: '11111111-1111-4111-8111-111111111111',
  client_request_id: '11111111-1111-4111-8111-111111111111',
  name: 'Durable routing evidence',
  description: '',
  status: 'completed',
  mode: 'replay',
  evidence_level: 'E0',
  track_evidence_levels: { routing: 'E0' },
  target_id: 'fixture',
  change_profile: 'recipe',
  suite_ids: ['evaluation-smoke'],
  track_ids: ['routing'],
  sample_limit: 4,
  concurrency: 1,
  seed: 42,
  progress: { percent: 100, completed: 1, total: 1 },
  created_at: '2026-08-30T00:00:00Z',
  completed_at: '2026-08-30T00:00:30Z',
}

const pairID = '22222222-2222-4222-8222-222222222222'
const pairCandidateID = '33333333-3333-4333-8333-333333333333'

function pairMember(
  role: 'baseline' | 'candidate',
  status: EvaluationRun['status'],
): EvaluationRun {
  const terminal = ['completed', 'failed', 'cancelled'].includes(status)
  return {
    ...run,
    id: role === 'baseline' ? run.id : pairCandidateID,
    client_request_id: role === 'baseline' ? run.id : pairCandidateID,
    name: role === 'baseline' ? 'Controlled baseline' : 'Controlled candidate',
    status,
    mode: 'live',
    baseline_run_id: role === 'candidate' ? run.id : undefined,
    controlled_pair: { pair_id: pairID, role },
    started_at: status === 'pending' ? undefined : run.created_at,
    completed_at: terminal ? run.completed_at : undefined,
  }
}

function pairExecution(
  baseline: EvaluationRun,
  candidate: EvaluationRun,
  state: EvaluationControlledPairState,
): EvaluationControlledPairExecution {
  return {
    schema_version: 'evaluation.v1',
    contract_version: 'evaluation-controlled-pair.v1',
    id: pairID,
    protocol: 'abba-interleaved.v1',
    baseline_source_run_id: '44444444-4444-4444-8444-444444444444',
    candidate_source_run_id: '55555555-5555-4555-8555-555555555555',
    baseline_run: baseline,
    candidate_run: candidate,
    state,
    capabilities:
      state === 'running'
        ? { can_cancel: true, can_delete: false }
        : { can_cancel: false, can_delete: true },
  }
}

function renderInspector(
  value: EvaluationRun | null,
  loading: boolean,
  error: string | null = null,
  options: {
    canRun?: boolean
    canDelete?: boolean
    mutationKey?: string | null
    controlledPairExecution?: EvaluationControlledPairExecution | null
    controlledPairLoading?: boolean
    controlledPairRefreshing?: boolean
    controlledPairError?: string | null
  } = {},
) {
  return renderToStaticMarkup(
    createElement(EvaluationRunInspector, {
      selectedRunID: run.id,
      run: value,
      loading,
      error,
      controlledPairExecution: options.controlledPairExecution ?? null,
      controlledPairLoading: options.controlledPairLoading ?? false,
      controlledPairRefreshing: options.controlledPairRefreshing ?? false,
      controlledPairError: options.controlledPairError ?? null,
      events: [],
      eventsConnected: false,
      eventsError: null,
      canRun: options.canRun ?? true,
      canDelete: options.canDelete ?? true,
      mutationKey: options.mutationKey ?? null,
      onRetry: () => undefined,
      onRetryControlledPair: () => undefined,
      onReconnectEvents: () => undefined,
      onStart: () => undefined,
      onCancel: () => undefined,
      onDelete: () => undefined,
      onOpenReport: () => undefined,
    }),
  )
}

function visibleText(markup: string): string {
  return markup
    .replace(/<[^>]*>/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()
}

describe('EvaluationRunInspector refresh continuity', () => {
  it('keeps the durable run visible while a detail refresh is in flight', () => {
    const markup = renderInspector(run, true)

    expect(markup).toContain('Durable routing evidence')
    expect(markup).toContain('Refreshing details…')
    expect(markup).toContain('Open report')
    expect(markup).not.toContain('Loading evaluation run')
    expect(markup).toContain('<dt>Evaluation scope</dt><dd>Diagnostic</dd>')
    expect(markup).toContain('aria-label="Selected evaluation run"')

    const text = visibleText(markup)
    expect(text).not.toMatch(/\bE[0-5]\b/)
    expect(text).not.toMatch(/\bG[0-9]\b/)
    expect(text).not.toContain(run.schema_version)
  })

  it('keeps raw run, baseline, and suite identifiers inside collapsed technical details', () => {
    const baselineID = '66666666-6666-4666-8666-666666666666'
    const linkedRun = { ...run, baseline_run_id: baselineID }
    const markup = renderInspector(linkedRun, false)
    const [summary, technicalDetails] = markup.split('<details')

    expect(summary).toContain('<dt>Baseline</dt><dd>Linked baseline</dd>')
    expect(summary).toContain('<dt>Benchmarks</dt><dd>1 selected</dd>')
    expect(summary).toContain('<dt>Evaluation target</dt><dd>Saved evaluation target</dd>')
    expect(summary).toContain('<dt>Workload</dt><dd>4 cases · 1 concurrent request</dd>')
    expect(summary).not.toContain(linkedRun.id)
    expect(summary).not.toContain(baselineID)
    expect(summary).not.toContain(linkedRun.target_id)
    expect(summary).not.toContain('evaluation-smoke')
    expect(technicalDetails).toContain('>Technical details</summary>')
    expect(technicalDetails).toContain(linkedRun.id)
    expect(technicalDetails).toContain(baselineID)
    expect(technicalDetails).toContain(linkedRun.target_id)
    expect(technicalDetails).toContain('evaluation-smoke')
    expect(technicalDetails).not.toMatch(/^ open(?:=|>)/)
  })

  it('uses the loading boundary only when no durable run is available', () => {
    const markup = renderInspector(null, true)

    expect(markup).toContain('Loading evaluation run')
    expect(markup).not.toContain('Refreshing details…')
  })

  it('keeps stale evidence inspectable when the latest detail refresh fails', () => {
    const backendMessage = 'decoder://run-detail E5 temporary-private-chain'
    const markup = renderInspector(run, false, backendMessage)

    expect(markup).toContain('Showing the last saved run details')
    expect(markup).toContain(backendMessage)
    expect(markup).toContain('Retry details')
    expect(markup).toContain('Durable routing evidence')
    expect(markup.indexOf(backendMessage)).toBeGreaterThan(
      markup.indexOf('data-evaluation-technical-details="true"'),
    )
  })

  it('uses aggregate capabilities when one member is terminal but the pair is running', () => {
    const baseline = pairMember('baseline', 'completed')
    const candidate = pairMember('candidate', 'running')
    const running = pairExecution(baseline, candidate, 'running')
    const terminalMember = renderInspector(baseline, false, null, {
      controlledPairExecution: running,
    })

    expect(terminalMember).toContain('>Open report<')
    expect(terminalMember).toContain('aria-label="Cancel controlled comparison"')
    expect(terminalMember).toContain('>Cancel comparison<')
    expect(terminalMember).not.toContain('>Start<')
    expect(terminalMember).not.toContain('>Delete comparison<')
  })

  it('exposes only authoritative aggregate lifecycle actions for controlled-pair members', () => {
    const baseline = pairMember('baseline', 'completed')
    const candidate = pairMember('candidate', 'completed')
    const terminal = renderInspector(baseline, false, null, {
      controlledPairExecution: pairExecution(baseline, candidate, 'terminal'),
    })
    expect(terminal).toContain('>Open report<')
    expect(terminal).toContain('aria-label="Delete controlled comparison"')
    expect(terminal).toContain('>Delete comparison<')
    expect(terminal).not.toContain('aria-label="Delete Controlled baseline"')

    const pendingBaseline = pairMember('baseline', 'pending')
    const pendingCandidate = pairMember('candidate', 'pending')
    const queued = renderInspector(pendingBaseline, false, null, {
      controlledPairExecution: pairExecution(pendingBaseline, pendingCandidate, 'pending'),
    })
    expect(queued).toContain('>Delete comparison<')
    expect(queued).not.toContain('>Start<')
  })

  it('honors permissions and aggregate mutation state for pair actions', () => {
    const pairedRun = pairMember('baseline', 'completed')
    const execution = pairExecution(pairedRun, pairMember('candidate', 'completed'), 'terminal')
    const readonly = renderInspector(pairedRun, false, null, {
      canRun: false,
      canDelete: false,
      controlledPairExecution: execution,
    })
    expect(readonly).toContain('>Open report<')
    expect(readonly).not.toContain('>Delete comparison<')

    const pending = renderInspector(pairedRun, false, null, {
      mutationKey: `delete-pair:${pairID}`,
      controlledPairExecution: execution,
    })
    expect(pending).toContain('>Deleting comparison…<')
    expect(pending).toContain('disabled=""')
  })

  it('withholds pair actions until the authoritative resource is available and owns its error', () => {
    const pairedRun = pairMember('baseline', 'completed')
    const loading = renderInspector(pairedRun, false, null, { controlledPairLoading: true })
    expect(loading).toContain('Loading comparison actions…')
    expect(loading).not.toContain('>Cancel comparison<')
    expect(loading).not.toContain('>Delete comparison<')

    const failed = renderInspector(pairedRun, false, null, {
      controlledPairError: 'worker://pair-read G8 temporary-private-chain',
    })
    expect(failed).toContain('Comparison actions could not be loaded')
    expect(failed).toContain('worker://pair-read G8 temporary-private-chain')
    expect(failed).toContain('Retry comparison actions')
    expect(failed).not.toContain('>Cancel comparison<')
    expect(failed).not.toContain('>Delete comparison<')
    expect(failed.indexOf('worker://pair-read G8 temporary-private-chain')).toBeGreaterThan(
      failed.indexOf('data-evaluation-technical-details="true"'),
    )
  })

  it('keeps terminal execution and empty-inspector request errors technical', () => {
    const executionError = 'worker://executor E4 panic=private-stack'
    const failedRun = renderInspector({ ...run, status: 'failed', error: executionError }, false)
    expect(failedRun).toContain('This run stopped before a report was published')
    expect(failedRun.indexOf(executionError)).toBeGreaterThan(
      failedRun.lastIndexOf('data-evaluation-technical-details="true"'),
    )

    const requestError = 'decoder://run-request E3 field=internal_only'
    const empty = renderInspector(null, false, requestError)
    expect(empty).toContain('Retry to load the selected run and its saved evidence')
    expect(empty.indexOf(requestError)).toBeGreaterThan(
      empty.lastIndexOf('data-evaluation-technical-details="true"'),
    )
    expect(empty).not.toContain('<details open')
  })
})
