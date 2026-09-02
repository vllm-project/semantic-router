import { describe, expect, it } from 'vitest'

import { createProbePlaygroundInvocation, isPlaygroundInvocation } from './playgroundInvocation'

const plan = {
  probe_id: 'private/zh-review',
  recipe_digest: 'sha256:recipe',
  recipe: 'private',
  editable: true,
  model: 'vllm-sr/mom-private',
  messages: [{ role: 'user', content: '请检查这个请求' }],
  tools: [{ type: 'function', function: { name: 'scan' } }],
  tool_choice: { type: 'function', function: { name: 'scan' } },
  request: {
    model: 'vllm-sr/mom-private',
    messages: [{ role: 'user', content: '请检查这个请求' }],
  },
}

describe('Playground probe invocation', () => {
  it('adds user intent without rewriting the fresh server run plan', () => {
    const invocation = createProbePlaygroundInvocation('edit', plan)

    expect(invocation).toEqual({
      version: 1,
      intent: 'edit',
      source: 'recipe-probe',
      probeId: plan.probe_id,
      recipeDigest: plan.recipe_digest,
      recipe: plan.recipe,
      editable: true,
      model: plan.model,
      messages: plan.messages,
      tools: plan.tools,
      toolChoice: plan.tool_choice,
      request: plan.request,
    })
    expect(isPlaygroundInvocation(invocation)).toBe(true)
  })

  it('rejects stale or incomplete navigation state', () => {
    expect(
      isPlaygroundInvocation({ ...createProbePlaygroundInvocation('run', plan), version: 2 }),
    ).toBe(false)
    expect(isPlaygroundInvocation({ version: 1, intent: 'run', source: 'recipe-probe' })).toBe(
      false,
    )
    expect(isPlaygroundInvocation(null)).toBe(false)
  })
})
