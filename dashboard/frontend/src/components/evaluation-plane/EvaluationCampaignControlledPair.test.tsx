import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import type { EvaluationControlledPairExecution } from '../../types/evaluationControlledPair'
import type { EvaluationCampaignReadiness } from '../../types/evaluationCampaign'
import type {
  EvaluationCatalogCampaignSlot,
  EvaluationCatalogChangeProfile,
  EvaluationRun,
} from '../../types/evaluationPlane'
import EvaluationCampaignControlledPair from './EvaluationCampaignControlledPair'

const useEvaluationControlledPairMock = vi.hoisted(() => vi.fn())

vi.mock('../../hooks/useEvaluationControlledPair', () => ({
  useEvaluationControlledPair: useEvaluationControlledPairMock,
}))

function productSurface(markup: string): string {
  return markup.replace(/\sdata-[\w-]+="[^"]*"/g, '')
}

const slot: EvaluationCatalogCampaignSlot = {
  gate_id: 'G3',
  name: 'Controlled paired-live value',
  description: 'Controlled comparison under a frozen policy.',
  disposition: 'required',
  binding_kind: 'controlled_pair',
  track_id: 'joint',
  mode: 'live',
  minimum_evidence_level: 'E4',
  accepted_executor_ids: ['live-runtime'],
}

const profile: EvaluationCatalogChangeProfile = {
  id: 'recipe',
  name: 'Routing recipe',
  description: 'Evaluate a routing recipe change.',
  campaign_slots: [slot],
}

const readiness: EvaluationCampaignReadiness = {
  schema_version: 'evaluation.v1',
  change_profile: profile.id,
  total_runs: 0,
  slots: [
    {
      gate_id: slot.gate_id,
      binding_kind: slot.binding_kind,
      eligible_run_ids: [],
      controlled_pair_source_run_ids: [],
      controlled_pair_candidate_run_ids: [],
      fidelity_reference_run_ids: [],
      fidelity_live_run_ids: [],
    },
  ],
}

function controlledRun(role: 'baseline' | 'candidate'): EvaluationRun {
  return {
    schema_version: 'evaluation.v1',
    id: role === 'baseline' ? 'baseline-run' : 'candidate-run',
    client_request_id: role === 'baseline' ? 'baseline-request' : 'candidate-request',
    name: role === 'baseline' ? 'Baseline' : 'Candidate',
    description: '',
    status: role === 'baseline' ? 'running' : 'failed',
    mode: 'live',
    evidence_level: 'E5',
    track_evidence_levels: { joint: 'E5' },
    target_id: role,
    change_profile: 'recipe',
    suite_ids: ['live-suite'],
    track_ids: ['joint'],
    sample_limit: 4,
    concurrency: 1,
    seed: 42,
    progress: {
      percent: role === 'baseline' ? 25 : 50,
      completed: role === 'baseline' ? 1 : 2,
      total: 4,
      message: `worker://${role}-progress E5 private-ledger-message`,
    },
    error: role === 'candidate' ? 'executor://candidate-failure private-stack' : undefined,
    created_at: '2026-09-01T00:00:00Z',
  }
}

function expectTechnicalOnly(markup: string, rawText: string) {
  const rawIndex = markup.indexOf(rawText)
  const detailsIndex = markup.lastIndexOf('<details', rawIndex)
  const detailsEnd = markup.indexOf('</details>', rawIndex)
  expect(rawIndex).toBeGreaterThan(-1)
  expect(detailsIndex).toBeGreaterThan(-1)
  expect(detailsEnd).toBeGreaterThan(rawIndex)
  expect(markup.slice(detailsIndex, rawIndex)).toContain('data-evaluation-technical-details="true"')
}

describe('EvaluationCampaignControlledPair', () => {
  beforeEach(() => {
    useEvaluationControlledPairMock.mockReturnValue({
      status: 'idle',
      execution: null,
      error: null,
      sourceIDs: null,
      create: vi.fn(),
      retry: vi.fn(),
      reset: vi.fn(),
    })
  })

  it('describes the controlled workflow without exposing contract codenames', () => {
    const markup = renderToStaticMarkup(
      createElement(EvaluationCampaignControlledPair, {
        runs: [],
        profile,
        slot,
        readiness,
        canCreate: true,
        disabled: false,
        activePairID: null,
        resumablePair: null,
        onProfileLockChange: () => undefined,
        onPairIdentityChange: () => undefined,
        onReady: () => undefined,
      }),
    )

    expect(markup).toContain('Controlled live comparison')
    expect(markup).toContain('order-balanced comparison')
    expect(markup).toContain('controlled value comparison')
    expect(markup).toContain('data-check-id="G3"')

    const surface = productSurface(markup)
    expect(surface).not.toMatch(/\b(?:E[0-5]|G[0-9])\b/)
    expect(surface).not.toMatch(/\b[a-z][a-z0-9_-]*(?:\.[a-z0-9_-]+)*\.v\d+\b/i)
  })

  it('keeps worker progress and workflow failures behind technical details', () => {
    const baseline = controlledRun('baseline')
    const candidate = controlledRun('candidate')
    const workflowError = 'decoder://controlled-pair G3 private-workflow-stack'
    const execution: EvaluationControlledPairExecution = {
      schema_version: 'evaluation.v1',
      contract_version: 'evaluation-controlled-pair.v1',
      id: 'pair-id',
      protocol: 'abba-interleaved.v1',
      baseline_source_run_id: 'baseline-source',
      candidate_source_run_id: 'candidate-source',
      baseline_run: baseline,
      candidate_run: candidate,
      state: 'terminal',
      capabilities: { can_cancel: false, can_delete: true },
    }
    useEvaluationControlledPairMock.mockReturnValue({
      status: 'error',
      execution,
      error: workflowError,
      sourceIDs: { baseline: 'baseline-source', candidate: 'candidate-source' },
      create: vi.fn(),
      retry: vi.fn(),
      reset: vi.fn(),
    })

    const markup = renderToStaticMarkup(
      createElement(EvaluationCampaignControlledPair, {
        runs: [],
        profile,
        slot,
        readiness,
        canCreate: true,
        disabled: false,
        activePairID: null,
        resumablePair: null,
        onProfileLockChange: () => undefined,
        onPairIdentityChange: () => undefined,
        onReady: () => undefined,
      }),
    )

    expect(markup).toContain('25% · Running')
    expect(markup).toContain('1 of 4 evaluation steps complete')
    expect(markup).toContain('One comparison run stopped before completing')
    for (const rawText of [
      baseline.progress.message || '',
      candidate.progress.message || '',
      candidate.error || '',
      workflowError,
    ]) {
      expectTechnicalOnly(markup, rawText)
    }
    expect(markup).not.toContain('<details open')
  })
})
