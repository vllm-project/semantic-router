import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'

import type { EvaluationCampaignGate } from '../../types/evaluationCampaign'
import { RecordedServiceNarrative } from './EvaluationCampaignDecisionTechnicalRecords'

describe('EvaluationCampaignDecisionTechnicalRecords', () => {
  it('retains signed service summary, check rationale, and recommendations verbatim', () => {
    const gates = [
      {
        id: 'G3',
        name: 'Controlled value comparison',
        disposition: 'required',
        verdict: 'unavailable',
        evidence_level: 'E5',
        source: 'campaign_slot',
        evidence_refs: [],
        rationale: 'G3 remains unavailable without server-owned production assignment evidence.',
      },
    ] satisfies EvaluationCampaignGate[]
    const markup = renderToStaticMarkup(
      createElement(RecordedServiceNarrative, {
        summary: 'Decision incomplete: one required campaign slot lacks qualified evidence.',
        gates,
        recommendations: ['Retain the sealed baseline and campaign anchors.'],
      }),
    )

    expect(markup).toContain('Recorded service narrative')
    expect(markup).toContain(
      'Decision incomplete: one required campaign slot lacks qualified evidence.',
    )
    expect(markup).toContain(
      'G3 remains unavailable without server-owned production assignment evidence.',
    )
    expect(markup).toContain('Retain the sealed baseline and campaign anchors.')
  })
})
