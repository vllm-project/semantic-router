import { createElement, createRef } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'

import type { EvaluationRun } from '../types/evaluationPlane'
import EvaluationRunActionDialogs from './EvaluationRunActionDialogs'

const pairID = '22222222-2222-4222-8222-222222222222'
const run: EvaluationRun = {
  schema_version: 'evaluation.v1',
  id: '11111111-1111-4111-8111-111111111111',
  client_request_id: '11111111-1111-4111-8111-111111111111',
  name: 'Candidate recipe',
  description: '',
  status: 'completed',
  mode: 'live',
  evidence_level: 'E1',
  track_evidence_levels: { routing: 'E1' },
  target_id: 'fixture',
  change_profile: 'recipe',
  suite_ids: ['evaluation-live'],
  track_ids: ['routing'],
  sample_limit: 4,
  concurrency: 1,
  seed: 42,
  progress: { percent: 100, completed: 1, total: 1 },
  controlled_pair: { pair_id: pairID, role: 'candidate' },
  created_at: '2026-08-30T00:00:00Z',
  completed_at: '2026-08-30T00:00:30Z',
}

function renderDialogs(options: {
  cancelTarget?: EvaluationRun | null
  deleteTarget?: EvaluationRun | null
  mutationKey?: string | null
  error?: string | null
}) {
  return renderToStaticMarkup(
    createElement(EvaluationRunActionDialogs, {
      cancelTarget: options.cancelTarget ?? null,
      deleteTarget: options.deleteTarget ?? null,
      mutationKey: options.mutationKey ?? null,
      error: options.error ?? null,
      returnFocusRef: createRef<HTMLElement>(),
      cancelReturnFocusMode: 'fallback',
      deleteReturnFocusMode: 'fallback',
      onCloseCancel: () => undefined,
      onCloseDelete: () => undefined,
      onConfirmCancel: () => undefined,
      onConfirmDelete: () => undefined,
    }),
  )
}

describe('EvaluationRunActionDialogs', () => {
  it('keeps the controlled-pair cancellation dialog pending as one aggregate action', () => {
    const backendError = 'worker://cancel-pair G3 private-stack'
    const markup = renderDialogs({
      cancelTarget: run,
      mutationKey: `cancel-pair:${pairID}`,
      error: backendError,
    })

    expect(markup.match(/role="alertdialog"/g)).toHaveLength(1)
    expect(markup).toContain('Cancel controlled comparison?')
    expect(markup).toContain('Both runs stop together.')
    expect(markup).toContain('aria-busy="true"')
    expect(markup).toContain('Cancelling comparison…')
    expect(markup).toContain('The controlled comparison could not be cancelled')
    const rawIndex = markup.indexOf(backendError)
    const detailsIndex = markup.lastIndexOf('<details', rawIndex)
    expect(detailsIndex).toBeGreaterThan(-1)
    expect(markup.slice(detailsIndex, rawIndex)).toContain(
      'data-evaluation-technical-details="true"',
    )
    expect(markup).not.toContain('<details open')
  })

  it('uses a stable product confirmation phrase instead of exposing the pair identifier', () => {
    const markup = renderDialogs({ deleteTarget: run })

    expect(markup.match(/role="alertdialog"/g)).toHaveLength(1)
    expect(markup).toContain('Delete controlled comparison?')
    expect(markup).toContain('Enter <strong>DELETE COMPARISON</strong> to confirm.')
    expect(markup).not.toContain(pairID)
  })
})
