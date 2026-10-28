import { describe, expect, it } from 'vitest'

import { buildEvaluationRecordsConfig } from './configPageEvaluationRecordsSupport'

describe('evaluation record configuration', () => {
  it('writes model-linked records to the unified top-level evaluation section', () => {
    const config = buildEvaluationRecordsConfig(
      { routing: { modelCards: [{ name: 'private-reasoner' }] } },
      [
        {
          model: ' private-reasoner ',
          benchmark: ' acme/support@1.0.0 ',
          reasoning_effort: ' high ',
          metrics: { resolution_rate: '0.82' },
        },
      ],
    )

    expect(config.evaluation?.records).toEqual([
      {
        model: 'private-reasoner',
        benchmark: 'acme/support@1.0.0',
        reasoning_effort: 'high',
        metrics: { resolution_rate: 0.82 },
        benchmark_profile: undefined,
        measured_at: undefined,
        metadata: undefined,
        source: undefined,
      },
    ])
    expect(config.routing?.modelCards?.[0]).toEqual({ name: 'private-reasoner' })
  })

  it('retains custom definitions when the operator clears every record', () => {
    const config = buildEvaluationRecordsConfig(
      {
        evaluation: {
          benchmarks: [
            {
              id: 'acme/support@1.0.0',
              display_name: 'Support',
              domain: 'support',
              default_profile: 'default',
              profiles: [],
              metrics: [],
            },
          ],
          records: [
            {
              model: 'private',
              benchmark: 'acme/support@1.0.0',
              metrics: { score: 1 },
            },
          ],
        },
      },
      [],
    )

    expect(config.evaluation?.benchmarks).toHaveLength(1)
    expect(config.evaluation?.records).toBeUndefined()
  })
})
