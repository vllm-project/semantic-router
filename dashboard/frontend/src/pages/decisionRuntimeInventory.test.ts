import { describe, expect, it } from 'vitest'
import { engineModelInventory } from './decisionRuntimeInventory'

describe('observed Engine model inventory', () => {
  it('retains native deployment identity and failure without inferring model readiness', () => {
    const result = engineModelInventory({
      data: [
        {
          id: 'judge',
          repo: 'provider/Decision',
          ready: false,
          status: 'failed',
          reason: 'load failed',
          surfaces: ['systemone'],
        },
        { id: 'missing-readiness', repo: 'provider/Decision' },
      ],
    })
    expect(result.deployments).toHaveLength(1)
    expect(result.deployments[0]).toMatchObject({
      name: 'judge',
      ready: false,
      state: 'failed',
      reason: 'load failed',
      repo: 'provider/Decision',
    })
  })
  it('does not interpret an unavailable controller response as an empty healthy inventory', () => {
    expect(() => engineModelInventory({ observed_mode: 'unknown' })).toThrow('unavailable')
  })
})
