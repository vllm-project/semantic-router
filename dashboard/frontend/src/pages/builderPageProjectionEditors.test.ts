import { describe, expect, it } from 'vitest'

import { projectionScoreToFields } from './builderPageProjectionEditors'

describe('Builder projection score fields', () => {
  it('keeps kb_metric knowledge base and metric without writing an empty name', () => {
    const fields = projectionScoreToFields({
      name: 'privacy_kb_bias',
      method: 'weighted_sum',
      inputs: [
        {
          signalType: 'kb_metric',
          signalName: '',
          kb: 'privacy_kb',
          metric: 'private_vs_public',
          weight: 1,
          valueSource: 'score',
        },
        { signalType: 'keyword', signalName: 'urgent', weight: 0.5 },
      ],
      pos: { Line: 1, Column: 1 },
    })

    expect(fields.inputs).toEqual([
      {
        type: 'kb_metric',
        kb: 'privacy_kb',
        metric: 'private_vs_public',
        weight: 1,
        value_source: 'score',
      },
      { type: 'keyword', name: 'urgent', weight: 0.5 },
    ])
  })
})
