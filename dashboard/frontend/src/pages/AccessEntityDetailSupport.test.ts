import { describe, expect, it, vi } from 'vitest'

import {
  createEntityPolicyNameResolver,
  resolveEntityPolicyNames,
} from './AccessEntityDetailSupport'

describe('Access entity policy names', () => {
  it('bounds concurrent detail lookups and reuses the modal-scoped cache', async () => {
    const pending: Array<{ finish: (name: string) => void }> = []
    let active = 0
    let maximumActive = 0
    const loadName = vi.fn(
      (_kind: 'access' | 'budget', _policyId: string) =>
        new Promise<string>((resolve) => {
          active += 1
          maximumActive = Math.max(maximumActive, active)
          pending.push({
            finish: (name) => {
              active -= 1
              resolve(name)
            },
          })
        }),
    )
    const resolver = createEntityPolicyNameResolver(loadName, 2)

    const first = resolver.resolve('access', 'group-1')
    const duplicate = resolver.resolve('access', 'group-1')
    const second = resolver.resolve('access', 'group-2')
    const third = resolver.resolve('access', 'group-3')

    expect(duplicate).toBe(first)
    expect(loadName).toHaveBeenCalledTimes(2)
    expect(maximumActive).toBe(2)

    pending.shift()?.finish('Engineering')
    await Promise.resolve()
    await Promise.resolve()

    expect(loadName).toHaveBeenCalledTimes(3)
    expect(maximumActive).toBe(2)
    pending.splice(0).forEach(({ finish }, index) => finish(`Group ${index + 2}`))

    await expect(Promise.all([first, duplicate, second, third])).resolves.toEqual([
      'Engineering',
      'Engineering',
      'Group 2',
      'Group 3',
    ])
    expect(loadName).toHaveBeenCalledTimes(3)
  })

  it('keeps input order and falls back to the policy id for failed or blank details', async () => {
    const loadName = vi.fn(async (_kind: 'access' | 'budget', policyId: string) => {
      if (policyId === 'missing') throw new Error('not found')
      if (policyId === 'blank') return '   '
      return `Name for ${policyId}`
    })
    const resolver = createEntityPolicyNameResolver(loadName)

    const names = await resolveEntityPolicyNames(
      'budget',
      ['budget-2', 'missing', 'blank', 'budget-2'],
      resolver,
    )

    expect([...names]).toEqual([
      ['budget-2', 'Name for budget-2'],
      ['missing', 'missing'],
      ['blank', 'blank'],
    ])
    expect(loadName).toHaveBeenCalledTimes(3)
  })
})
