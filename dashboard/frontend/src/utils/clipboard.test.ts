import { afterEach, describe, expect, it, vi } from 'vitest'

import { copyText } from './clipboard'

describe('copyText', () => {
  afterEach(() => {
    vi.restoreAllMocks()
    vi.unstubAllGlobals()
  })

  it('copies through the selection fallback on an HTTP page', async () => {
    const setData = vi.fn()
    let copyListener: ((event: ClipboardEvent) => void) | undefined
    const command = vi.fn(() => {
      copyListener?.({
        clipboardData: { setData },
        preventDefault: vi.fn(),
      } as unknown as ClipboardEvent)
      return true
    })
    const textarea = {
      value: '',
      readOnly: false,
      style: {} as Record<string, string>,
      focus: vi.fn(),
      select: vi.fn(),
      setSelectionRange: vi.fn(),
      remove: vi.fn(),
    }
    const documentStub = {
      body: { appendChild: vi.fn() },
      activeElement: null,
      createElement: vi.fn(() => textarea),
      execCommand: command,
      addEventListener: vi.fn((_name, listener) => {
        copyListener = listener as (event: ClipboardEvent) => void
      }),
      removeEventListener: vi.fn(),
    }
    vi.stubGlobal('window', { isSecureContext: false })
    vi.stubGlobal('navigator', {})
    vi.stubGlobal('document', documentStub)

    await expect(copyText('personal invitation')).resolves.toBe(true)
    expect(command).toHaveBeenCalledWith('copy')
    expect(setData).toHaveBeenCalledWith('text/plain', 'personal invitation')
    expect(textarea.remove).toHaveBeenCalledOnce()
  })

  it('does not report success when the browser accepts the command without writing data', async () => {
    const textarea = {
      value: '',
      readOnly: false,
      style: {} as Record<string, string>,
      focus: vi.fn(),
      select: vi.fn(),
      setSelectionRange: vi.fn(),
      remove: vi.fn(),
    }
    vi.stubGlobal('window', { isSecureContext: false })
    vi.stubGlobal('navigator', {})
    vi.stubGlobal('document', {
      body: { appendChild: vi.fn() },
      activeElement: null,
      createElement: vi.fn(() => textarea),
      execCommand: vi.fn(() => true),
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
    })

    await expect(copyText('not actually copied')).resolves.toBe(false)
  })
})
