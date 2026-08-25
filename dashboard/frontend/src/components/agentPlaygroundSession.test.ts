import { describe, expect, it } from 'vitest'

import { buildAgentSessionInput } from './agentPlaygroundSession'

describe('buildAgentSessionInput', () => {
  it('binds a Mixture-of-Models session to the selected key and its team context', () => {
    expect(
      buildAgentSessionInput({
        keyId: '11111111-1111-4111-8111-111111111111',
        effectiveTeamId: '22222222-2222-4222-8222-222222222222',
        mode: 'builder',
        model: { id: 'vllm-sr/blend', description: 'Blend' },
        title: 'Tune Blend',
      }),
    ).toEqual({
      keyId: '11111111-1111-4111-8111-111111111111',
      effectiveTeamId: '22222222-2222-4222-8222-222222222222',
      mode: 'builder',
      target: { kind: 'entrypoint', id: 'vllm-sr/blend' },
      title: 'Tune Blend',
    })
  })

  it('binds a direct Model session without inventing a team context', () => {
    expect(
      buildAgentSessionInput({
        keyId: '33333333-3333-4333-8333-333333333333',
        mode: 'chat',
        model: { id: 'local/coder', description: 'Coder', kind: 'individual' },
        title: 'Test Coder',
      }),
    ).toEqual({
      keyId: '33333333-3333-4333-8333-333333333333',
      mode: 'chat',
      target: { kind: 'model', id: 'local/coder' },
      title: 'Test Coder',
    })
  })
})
