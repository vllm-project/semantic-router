import { describe, expect, it } from 'vitest'

import type { PreparationBenchmark } from './datasetPreparationApi'
import { filterPreparationBenchmarks, largestPreparationSize } from './prepareDialogSupport'

const benchmark = (
  id: string,
  name: string,
  profiles: PreparationBenchmark['profiles'] = { smoke: 5, quick: 100, standard: 500 },
): PreparationBenchmark => ({
  id,
  name,
  profiles,
  source_url: `https://example.com/${id}`,
  access_note: '',
  dependencies: [],
})

const catalog = [
  benchmark('simpleqa-verified', 'SimpleQA Verified'),
  benchmark('mmlu-pro', 'MMLU-Pro', { smoke: 14, quick: 500, standard: 2000 }),
  benchmark('hle', "Humanity's Last Exam (text)"),
]

describe('filterPreparationBenchmarks', () => {
  it('keeps the whole catalog for an empty or blank query', () => {
    expect(filterPreparationBenchmarks(catalog, '')).toBe(catalog)
    expect(filterPreparationBenchmarks(catalog, '   ')).toBe(catalog)
  })

  it('matches the name or the id without regard to case', () => {
    expect(filterPreparationBenchmarks(catalog, 'mmlu').map((item) => item.id)).toEqual([
      'mmlu-pro',
    ])
    expect(filterPreparationBenchmarks(catalog, ' LAST exam ').map((item) => item.id)).toEqual([
      'hle',
    ])
    expect(
      filterPreparationBenchmarks(catalog, 'simpleqa-verified').map((item) => item.id),
    ).toEqual(['simpleqa-verified'])
    expect(filterPreparationBenchmarks(catalog, 'gpqa')).toEqual([])
  })
})

describe('largestPreparationSize', () => {
  it('reads the largest size whatever order the sizes grow in', () => {
    expect(largestPreparationSize(catalog[1])).toBe(2000)
    expect(
      largestPreparationSize(benchmark('odd', 'Odd', { smoke: 40, quick: 12, standard: 30 })),
    ).toBe(40)
  })

  it('ignores a size the service did not report', () => {
    const partial = benchmark('partial', 'Partial', {
      smoke: 3,
      quick: Number.NaN,
      standard: undefined as unknown as number,
    })
    expect(largestPreparationSize(partial)).toBe(3)
  })
})
