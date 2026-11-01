import { describe, expect, it } from 'vitest'

import type { RouteModelInput } from '@/lib/dslMutations'

import { routeInputHasErrors, validateRouteInput } from './builderPageRoutePreview'

describe('visual route editor ceiling validation', () => {
  it('blocks save and create while a ModelRef ceiling is invalid', () => {
    const invalidModels: RouteModelInput[] = [{ model: 'qwen', maxCompletionTokens: 0 }]

    expect(validateRouteInput('math', invalidModels, undefined, [])).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          level: 'error',
          message: expect.stringContaining('max_completion_tokens must be a finite integer >= 1'),
        }),
      ]),
    )
    expect(routeInputHasErrors('math', invalidModels, undefined, [])).toBe(true)
    expect(
      routeInputHasErrors('math', [{ model: 'qwen', maxCompletionTokens: 64 }], undefined, []),
    ).toBe(false)
  })
})
