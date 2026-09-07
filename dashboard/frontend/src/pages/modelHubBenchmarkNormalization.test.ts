import { describe, expect, it } from 'vitest'

import type { CatalogBenchmarkMetric, CatalogMetricNormalization } from '../types/modelCatalog'
import {
  modelHubBenchmarkNormalizedValue,
  modelHubBenchmarkRawValueLabel,
  modelHubBenchmarkValueLabel,
} from './modelHubBenchmarkNormalization'

const metric = (normalization: CatalogMetricNormalization): CatalogBenchmarkMetric => ({
  id: 'score',
  unit: 'score',
  direction: 'higher_is_better',
  range: [-3000, 3000],
  normalization,
})

describe('model hub benchmark normalization', () => {
  it.each([
    ['identity', 0.75, { type: 'identity' }, 0.75],
    ['one minus', 0.25, { type: 'one_minus' }, 0.75],
    ['linear clamp', 1769.1, { type: 'linear_clamp', min: 500, max: 2500 }, 0.63455],
    [
      'piecewise linear',
      15,
      {
        type: 'piecewise_linear',
        points: [
          { input: 10, output: 0.2 },
          { input: 20, output: 0.8 },
        ],
      },
      0.5,
    ],
    ['logistic', 0, { type: 'logistic', k: 1, x0: 0 }, 0.5],
    ['lookup', 7, { type: 'lookup', values: { '7': 0.9 } }, 0.9],
  ] as Array<[string, number, CatalogMetricNormalization, number]>)(
    'applies %s',
    (_name, value, normalization, expected) => {
      expect(modelHubBenchmarkNormalizedValue(value, metric(normalization))).toBeCloseTo(expected)
    },
  )

  it('formats normalized output as percent while preserving the raw unit label', () => {
    const elo = metric({ type: 'linear_clamp', min: 500, max: 2500 })

    expect(modelHubBenchmarkValueLabel(1769.1, elo)).toBe('63.5%')
    expect(modelHubBenchmarkRawValueLabel(1769.1, elo)).toBe('1769.10 score')
  })
})
