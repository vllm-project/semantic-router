import { describe, expect, it } from 'vitest'
import { getSignalFieldSchema } from './dslSchemas'
import { requiredStructuredFieldErrors } from '../pages/builderPageStructuredFieldSupport'

describe('embedding candidate bank authoring', () => {
  const fields = getSignalFieldSchema('embedding')
  it('supports image-only positives and both negative modalities', () => {
    for (const key of [
      'candidates',
      'image_candidates',
      'negative_candidates',
      'negative_image_candidates',
    ]) {
      expect(fields.find((field) => field.key === key)).toMatchObject({ type: 'string[]' })
    }
    expect(
      requiredStructuredFieldErrors(fields, {
        threshold: -0.1,
        image_candidates: ['./positive.png'],
        negative_image_candidates: ['./negative.png'],
      }),
    ).toEqual([])
  })
  it('explains the signed margin threshold range', () => {
    expect(fields.find((field) => field.key === 'threshold')).toMatchObject({ min: -2, max: 2 })
  })
})
