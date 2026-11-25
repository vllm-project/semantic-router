import { describe, expect, it } from 'vitest'
import { selectedQuestionCount } from './datasetPresentation'
import type { DatasetSelection } from './types'

type Entry = DatasetSelection['benchmarks'][number]

const entry = (id: string, overrides: Partial<Entry> = {}): Entry => ({
  id,
  title: id,
  eligible: true,
  case_count: 14,
  source_ids: [`${id}-smoke`],
  reason: null,
  reason_code: null,
  ...overrides,
})

const notPrepared = (id: string) =>
  entry(id, {
    eligible: false,
    case_count: 0,
    source_ids: [],
    reason: 'Prepared automatically',
    reason_code: 'not_prepared',
  })

describe('selectedQuestionCount', () => {
  it('sums prepared counts for the selected benchmarks', () => {
    const selection = { benchmarks: [entry('mmlu-pro'), entry('gpqa-diamond', { case_count: 10 })] }
    expect(selectedQuestionCount(selection, ['mmlu-pro'])).toBe(14)
    expect(selectedQuestionCount(selection, ['mmlu-pro', 'gpqa-diamond'])).toBe(24)
  })

  it('is unknown, not 0, while a selected benchmark is still to be prepared', () => {
    const selection = { benchmarks: [notPrepared('mmlu-pro'), entry('gpqa-diamond')] }
    expect(selectedQuestionCount(selection, ['mmlu-pro'])).toBeNull()
    expect(selectedQuestionCount(selection, ['mmlu-pro', 'gpqa-diamond'])).toBeNull()
  })

  it('is unknown when nothing is selected', () => {
    expect(selectedQuestionCount({ benchmarks: [entry('mmlu-pro')] }, [])).toBeNull()
  })
})
