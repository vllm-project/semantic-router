import { describe, expect, it } from 'vitest'

import { resolveConnectedModelName } from './configPageConnectModelSupport'

describe('connected model naming', () => {
  it('keeps the upstream model id when the public namespace is free', () => {
    expect(resolveConnectedModelName('', 'vllm', 'local/qwen', new Set())).toBe('local/qwen')
  })

  it('scopes an upstream model id that conflicts with a virtual model', () => {
    const reserved = new Set(['vllm-sr/mom-v1-blend'])

    expect(resolveConnectedModelName('', 'vllm', 'vllm-sr/mom-v1-blend', reserved)).toBe(
      'vllm/vllm-sr/mom-v1-blend',
    )
  })

  it('creates a stable unique name when the provider-scoped name is also occupied', () => {
    const reserved = new Set(['blend', 'vllm/blend', 'vllm/blend-2'])

    expect(resolveConnectedModelName('', 'vllm', 'blend', reserved)).toBe('vllm/blend-3')
  })
})
