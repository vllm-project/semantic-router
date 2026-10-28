import { describe, expect, it } from 'vitest'

import { requiredStructuredFieldErrors } from '../pages/builderPageStructuredFieldSupport'
import { getSignalFieldSchema } from './dslSchemas'

describe('complexity signal field schema', () => {
  const fields = getSignalFieldSchema('complexity')
  const byKey = (key: string) => fields.find((field) => field.key === key)
  const boundaryKeys = ['hard_above', 'easy_below', 'hard_below', 'easy_above']

  it('exposes both boundary pairs alongside the local threshold', () => {
    expect(fields.map((field) => field.key)).toEqual(
      expect.arrayContaining(['threshold', ...boundaryKeys, 'hard', 'easy', 'composer']),
    )
    for (const key of boundaryKeys) {
      expect(byKey(key)?.type, key).toBe('number')
      expect(byKey(key)?.description, key).toBeTruthy()
    }
  })

  it('does not require threshold: a score.v1 rule states a boundary pair instead', () => {
    expect(byKey('threshold')?.required).toBeFalsy()
    for (const key of boundaryKeys) {
      expect(byKey(key)?.required, key).toBeFalsy()
    }
  })

  it('does not require candidates: a remote backend never reads them', () => {
    expect(byKey('hard')?.required).toBeFalsy()
    expect(byKey('easy')?.required).toBeFalsy()
  })

  it('lets a remote-only rule through the required-field check', () => {
    expect(requiredStructuredFieldErrors(fields, { hard_above: 0.85, easy_below: 0.6 })).toEqual([])
    expect(requiredStructuredFieldErrors(fields, { hard_below: 0.2, easy_above: 0.6 })).toEqual([])
  })
})
