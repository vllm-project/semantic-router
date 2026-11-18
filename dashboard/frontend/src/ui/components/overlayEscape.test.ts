import { describe, expect, it } from 'vitest'

import { escapeBelongsToNestedPopup } from './overlayEscape'

const element = (connected: boolean, inside: string | null) => ({
  isConnected: connected,
  closest: (selector: string) => (inside && selector.includes(inside) ? {} : null),
})

describe('escapeBelongsToNestedPopup', () => {
  it('leaves an Escape inside the dialog itself to the dialog', () => {
    expect(
      escapeBelongsToNestedPopup({ target: element(true, null) as unknown as EventTarget }),
    ).toBe(false)
    expect(escapeBelongsToNestedPopup({ target: null })).toBe(false)
  })

  it('claims an Escape from an open list or menu, or from a popup that just unmounted', () => {
    expect(
      escapeBelongsToNestedPopup({ target: element(true, 'listbox') as unknown as EventTarget }),
    ).toBe(true)
    expect(
      escapeBelongsToNestedPopup({ target: element(true, 'menu') as unknown as EventTarget }),
    ).toBe(true)
    expect(
      escapeBelongsToNestedPopup({ target: element(false, null) as unknown as EventTarget }),
    ).toBe(true)
  })
})
