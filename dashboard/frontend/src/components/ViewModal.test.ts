import { describe, expect, it, vi } from 'vitest'

import { transitionFromViewToEdit } from './ViewModal'

describe('ViewModal edit transition', () => {
  it('closes the view before opening edit', () => {
    const calls: string[] = []
    const onClose = vi.fn(() => calls.push('close'))
    const onEdit = vi.fn(() => calls.push('edit'))

    transitionFromViewToEdit(onClose, onEdit)

    expect(calls).toEqual(['close', 'edit'])
    expect(onClose).toHaveBeenCalledOnce()
    expect(onEdit).toHaveBeenCalledOnce()
  })
})