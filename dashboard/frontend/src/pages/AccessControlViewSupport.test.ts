import { describe, expect, it } from 'vitest'
import type { AccessAPIKey } from '../utils/inferenceAccessApi'
import { keyPolicy } from './AccessControlViewSupport'

const inheritedKey: AccessAPIKey = {
  id: 'key-1',
  name: 'Taylor',
  prefix: 'vsr_example',
  contextTeamId: 'team-1',
  ownerType: 'user',
  ownerId: 'user-1',
  status: 'active',
  accessGroupIds: [],
  effectiveAccess: [
    { resourceType: 'entrypoint', resourceId: 'blend' },
    { resourceType: 'model', resourceId: 'qwen' },
  ],
}

describe('keyPolicy', () => {
  it('uses exact effective resources returned by the Router', () => {
    expect(keyPolicy(inheritedKey, [])).toEqual({
      direct: false,
      resources: [
        { resourceType: 'entrypoint', resourceId: 'blend' },
        { resourceType: 'model', resourceId: 'qwen' },
      ],
    })
  })

  it('labels effective resources as direct when the key has an explicit group', () => {
    expect(keyPolicy({ ...inheritedKey, accessGroupIds: ['group-1'] }, [])).toEqual({
      direct: true,
      resources: [
        { resourceType: 'entrypoint', resourceId: 'blend' },
        { resourceType: 'model', resourceId: 'qwen' },
      ],
    })
  })
})
