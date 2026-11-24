import { describe, expect, it } from 'vitest'
import { mergeChips } from './chip-input-support'

describe('mergeChips', () => {
  it('adds trimmed values and skips blanks and duplicates, ignoring case by default', () => {
    expect(mergeChips(['Contract'], [' clause ', '', 'contract', 'Clause'])).toEqual({
      next: ['Contract', 'clause'],
      error: null,
    })
  })

  it('keeps case-distinct values when case sensitive', () => {
    expect(mergeChips(['GPU'], ['gpu'], { caseSensitive: true }).next).toEqual(['GPU', 'gpu'])
  })

  it('refuses the whole batch at the first invalid value', () => {
    const validate = (candidate: string) =>
      candidate.length > 5 ? 'Keep keywords under six characters' : null
    expect(mergeChips(['a'], ['bb', 'too-long', 'cc'], { validate })).toEqual({
      next: ['a'],
      error: 'Keep keywords under six characters',
    })
  })
})
