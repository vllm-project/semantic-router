import { describe, expect, it } from 'vitest'

import type { ASTModelRef } from '@/types/dsl'

import { astModelToInput, updateRoute, type RouteModelInput } from './dslMutations'

describe('visual route editor ModelRef ceilings', () => {
  it('round-trips maxCompletionTokens through parse, edit, and save', () => {
    const source = `ROUTE math {
  PRIORITY 50

  MODEL "qwen" (lora = "adapter-a", max_completion_tokens = 1024)
}
`

    const parsed: ASTModelRef = {
      model: 'qwen',
      lora: 'adapter-a',
      maxCompletionTokens: 1024,
      pos: { Line: 4, Column: 3 },
    }
    const loaded = astModelToInput(parsed)
    expect(loaded.maxCompletionTokens).toBe(1024)

    const edited: RouteModelInput[] = [{ ...loaded, maxCompletionTokens: 256 }]
    const saved = updateRoute(source, 'math', {
      priority: 50,
      models: edited,
      plugins: [],
    })
    expect(saved).toContain('max_completion_tokens = 256')
    expect(saved).toContain('lora = "adapter-a"')

    const resaved = updateRoute(source, 'math', {
      priority: 50,
      models: [loaded],
      plugins: [],
    })
    expect(resaved).toContain('max_completion_tokens = 1024')
  })

  it('omits an unset ceiling and rejects a non-positive value from the serializer', () => {
    const source = `ROUTE math {
  PRIORITY 10

  MODEL "qwen" (max_completion_tokens = 64)
}
`
    const withoutCeiling = updateRoute(source, 'math', {
      priority: 10,
      models: [{ model: 'qwen' }, { model: 'llama', maxCompletionTokens: 128 }],
      plugins: [],
    })
    expect(withoutCeiling).toContain('MODEL "qwen",')
    expect(withoutCeiling).toContain('"llama" (max_completion_tokens = 128)')
    expect(withoutCeiling).not.toMatch(/"qwen"[^\n]*max_completion_tokens/)

    const droppedInvalid = updateRoute(source, 'math', {
      priority: 10,
      models: [{ model: 'qwen', maxCompletionTokens: 0 }],
      plugins: [],
    })
    expect(droppedInvalid).not.toContain('max_completion_tokens')
  })
})
