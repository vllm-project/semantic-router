import { describe, expect, it } from 'vitest'
import type { AccessAPIKey } from '../utils/inferenceAccessApi'
import { keyPolicy } from './AccessControlViewSupport'

const inheritedKey: AccessAPIKey = {
  id: 'key-1',
  name: 'Taylor',
  prefix: 'vsr_example',
  userId: 'user-1',
  effectiveTeamId: 'team-1',
  status: 'active',
  accessGroupIds: [],
  modelPatterns: ['vllm-sr/mom-v1-blend', 'vllm-sr/mom-v1-blend', 'local/qwen-*'],
}

describe('keyPolicy', () => {
  it('uses effective model patterns returned by the self-service API', () => {
    expect(keyPolicy(inheritedKey, [])).toEqual({
      direct: false,
      patterns: ['vllm-sr/mom-v1-blend', 'local/qwen-*'],
    })
  })

  it('labels effective patterns as direct when the key has an explicit group', () => {
    expect(keyPolicy({ ...inheritedKey, accessGroupIds: ['group-1'] }, [])).toEqual({
      direct: true,
      patterns: ['vllm-sr/mom-v1-blend', 'local/qwen-*'],
    })
  })
})
