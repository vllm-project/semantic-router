import { describe, expect, it } from 'vitest'

import { projectionScoreToFields } from './builderPageProjectionEditors'

describe('Builder projection score fields', () => {
  it('writes each input by its signal type and name', () => {
    const fields = projectionScoreToFields({
      name: 'urgency',
      method: 'weighted_sum',
      inputs: [
        { signalType: 'keyword', signalName: 'urgent', weight: 0.5, valueSource: 'binary' },
        { signalType: 'projection', signalName: 'effort', weight: 1 },
      ],
      pos: { Line: 1, Column: 1 },
    })

    expect(fields.inputs).toEqual([
      { type: 'keyword', name: 'urgent', weight: 0.5, value_source: 'binary' },
      { type: 'projection', name: 'effort', weight: 1 },
    ])
  })
})
