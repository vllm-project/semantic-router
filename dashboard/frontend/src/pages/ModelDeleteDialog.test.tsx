import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'

import { modelDeleteConfirmation } from './configPageModelInventory'

describe('ModelDeleteDialog', () => {
  it('asks for a typed phrase only when several models are deleted at once', () => {
    expect(modelDeleteConfirmation(1)).toBe('')
    expect(modelDeleteConfirmation(2)).toBe('DELETE 2')
    expect(modelDeleteConfirmation(14)).toBe('DELETE 14')
  })

  it('is an accessible alert dialog that blocks implicit dismissal after typed input or while pending', () => {
    const source = readFileSync(new URL('./ModelDeleteDialog.tsx', import.meta.url), 'utf8')

    expect(source).toContain('role="alertdialog"')
    expect(source).toContain('aria-busy={pending}')
    expect(source).toContain('const dismissible = !pending && confirmation.length === 0')
    expect(source).toContain('if (!open && dismissible) onCancel()')
    expect(source).toContain('if (!dismissible) event.preventDefault()')
    expect(source).toContain('data-dialog-initial-focus')
    expect(source).toContain('disabled={pending || !confirmationReady}')
  })
})
