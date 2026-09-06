import { describe, expect, it } from 'vitest'

import {
  parseEvaluationRoute,
  removeEvaluationRun,
  serializeEvaluationRoute,
} from './evaluationRoute'

const CONTROLLED_PAIR_ID = '11111111-1111-4111-8111-111111111111'
const CONTROLLED_PAIR_PROFILE = 'model_pool' as const

describe('evaluation route contract', () => {
  it('keeps only state owned by the active workspace', () => {
    expect(
      parseEvaluationRoute(
        new URLSearchParams(
          'view=reports&report=report-1&run=stale&baseline=stale&candidate=stale',
        ),
      ),
    ).toEqual({
      view: 'reports',
      reportRunID: 'report-1',
      controlledPairID: null,
      controlledPairProfileID: null,
    })
    expect(
      serializeEvaluationRoute({
        view: 'reports',
        reportRunID: 'report-1',
        controlledPairID: null,
        controlledPairProfileID: null,
      }).toString(),
    ).toBe('view=reports&report=report-1')
  })

  it('normalizes unknown workspaces to overview', () => {
    expect(parseEvaluationRoute(new URLSearchParams('view=unknown&report=ignored'))).toEqual({
      view: 'overview',
      controlledPairID: null,
      controlledPairProfileID: null,
    })
    expect(
      serializeEvaluationRoute({
        view: 'overview',
        controlledPairID: null,
        controlledPairProfileID: null,
      }).toString(),
    ).toBe('')
  })

  it('round-trips a public Mixture entrypoint without accepting an execution address', () => {
    const route = parseEvaluationRoute(
      new URLSearchParams('view=new&entrypoint=vllm-sr%2Fauto&target_url=http%3A%2F%2Fprivate'),
    )
    expect(route).toEqual({
      view: 'new',
      entrypoint: 'vllm-sr/auto',
      controlledPairID: null,
      controlledPairProfileID: null,
    })
    expect(serializeEvaluationRoute(route).toString()).toBe('view=new&entrypoint=vllm-sr%2Fauto')
  })

  it('clears a deleted run only from the workspace that owns it', () => {
    expect(
      removeEvaluationRun(
        {
          view: 'compare',
          baselineRunID: 'baseline',
          candidateRunID: 'candidate',
          campaignID: 'campaign',
          controlledPairID: CONTROLLED_PAIR_ID,
          controlledPairProfileID: CONTROLLED_PAIR_PROFILE,
        },
        'candidate',
      ),
    ).toEqual({
      view: 'compare',
      baselineRunID: null,
      candidateRunID: null,
      campaignID: 'campaign',
      controlledPairID: CONTROLLED_PAIR_ID,
      controlledPairProfileID: CONTROLLED_PAIR_PROFILE,
    })
  })

  it('round-trips the immutable campaign decision independently from a diagnostic pair', () => {
    const route = parseEvaluationRoute(
      new URLSearchParams('view=compare&campaign=campaign-1&baseline=baseline&candidate=candidate'),
    )
    expect(route).toEqual({
      view: 'compare',
      baselineRunID: 'baseline',
      candidateRunID: 'candidate',
      campaignID: 'campaign-1',
      controlledPairID: null,
      controlledPairProfileID: null,
    })
    expect(serializeEvaluationRoute(route).toString()).toBe(
      'view=compare&baseline=baseline&candidate=candidate&campaign=campaign-1',
    )
  })

  it('preserves a canonical controlled-pair workflow across workspaces and rejects invalid ids', () => {
    const reports = parseEvaluationRoute(
      new URLSearchParams(
        `view=reports&report=report-1&controlled_pair=${CONTROLLED_PAIR_ID}&controlled_pair_profile=${CONTROLLED_PAIR_PROFILE}`,
      ),
    )
    expect(reports.controlledPairID).toBe(CONTROLLED_PAIR_ID)
    expect(reports.controlledPairProfileID).toBe(CONTROLLED_PAIR_PROFILE)
    expect(serializeEvaluationRoute(reports).toString()).toBe(
      `controlled_pair=${CONTROLLED_PAIR_ID}&controlled_pair_profile=${CONTROLLED_PAIR_PROFILE}&view=reports&report=report-1`,
    )
    expect(
      parseEvaluationRoute(
        new URLSearchParams('view=compare&controlled_pair=not-a-canonical-identity'),
      ).controlledPairID,
    ).toBeNull()
    expect(
      parseEvaluationRoute(
        new URLSearchParams(`view=compare&controlled_pair=${CONTROLLED_PAIR_ID}`),
      ),
    ).toMatchObject({ controlledPairID: null, controlledPairProfileID: null })
    expect(
      parseEvaluationRoute(
        new URLSearchParams(
          `view=compare&controlled_pair=${CONTROLLED_PAIR_ID}&controlled_pair_profile=not%20portable`,
        ),
      ),
    ).toMatchObject({ controlledPairID: null, controlledPairProfileID: null })
  })
})
