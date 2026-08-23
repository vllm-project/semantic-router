import { describe, expect, it, vi } from 'vitest'

import { filterViewActionsForMode } from './configPageRouterSectionSupport'

describe('config detail action visibility', () => {
  const viewAction = {
    label: 'View topology',
    onClick: vi.fn(),
    availableWhenReadonly: true,
  }
  const editAction = { label: 'Delete model', onClick: vi.fn(), tone: 'destructive' as const }

  it('keeps explicitly read-safe actions visible in readonly mode', () => {
    expect(filterViewActionsForMode([viewAction, editAction], true)).toEqual([viewAction])
  })

  it('keeps every supplied action in write mode', () => {
    expect(filterViewActionsForMode([viewAction, editAction], false)).toEqual([
      viewAction,
      editAction,
    ])
  })
})
