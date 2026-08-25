import { describe, expect, it, vi } from 'vitest'

import type { SelfInferenceKey } from './routerManagementTypes'
import {
  mergeSelfInferenceKeyPages,
  restoreSelfInferenceKeySelection,
  selfInferenceKeyListQuery,
} from './selfInferenceKeys'

const key = (id: string, name = id): SelfInferenceKey => ({
  keyId: id,
  name,
  owner: { type: 'user', id: '11111111-1111-4111-8111-111111111111' },
})

describe('self inference key selection', () => {
  it('restores a saved key beyond the bounded first page through scoped by-ID lookup', async () => {
    const saved = key('22222222-2222-4222-8222-222222222222', 'Saved key')
    const lookup = vi.fn().mockResolvedValue(saved)

    await expect(
      restoreSelfInferenceKeySelection([key('first')], saved.keyId, lookup),
    ).resolves.toEqual(saved)
    expect(lookup).toHaveBeenCalledWith(saved.keyId)
  })

  it('falls back without exposing a key when scoped lookup denies or cannot find it', async () => {
    const first = key('first')
    const lookup = vi.fn().mockRejectedValue(new Error('not found'))

    await expect(
      restoreSelfInferenceKeySelection([first], 'out-of-scope', lookup),
    ).resolves.toEqual(first)
    await expect(restoreSelfInferenceKeySelection([], 'out-of-scope', lookup)).resolves.toBeNull()
  })

  it('bounds search pages and loaded DOM options', () => {
    expect(
      selfInferenceKeyListQuery({
        cursor: 'opaque',
        search: `  ${'x'.repeat(250)}  `,
        pageSize: 200,
      }).toString(),
    ).toBe(`pageSize=25&search=${'x'.repeat(200)}&cursor=opaque`)
    expect(
      mergeSelfInferenceKeyPages(
        Array.from({ length: 80 }, (_, index) => key(`old-${index}`)),
        Array.from({ length: 40 }, (_, index) => key(`new-${index}`)),
      ),
    ).toHaveLength(100)
  })
})
