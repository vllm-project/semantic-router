import { describe, expect, it, vi } from 'vitest'

// Capture the listener the store installs at module load against a stub window,
// so the guard is exercised in Node without a DOM environment.
const registered = new Map<
  string,
  (event: { preventDefault: () => void; returnValue: string }) => void
>()
vi.stubGlobal('window', {
  addEventListener: (
    type: string,
    fn: (event: { preventDefault: () => void; returnValue: string }) => void,
  ) => {
    registered.set(type, fn)
  },
})

vi.mock('@/lib/wasm', () => ({
  wasmBridge: {
    init: vi.fn().mockResolvedValue(undefined),
    decompile: vi.fn(),
    format: vi.fn(),
    compile: vi.fn(),
    validate: vi.fn(),
    parseAST: vi.fn(),
  },
}))

// The import performs the module-level listener installation. Reset the module
// registry first so another file having loaded the store cannot serve a cached
// instance that never saw the stub window.
vi.resetModules()
const { useDSLStore } = await import('./dslStore')
const { initialDSLState } = await import('./dslStoreSupport')

// The capture is complete; the stub window is no longer needed.
vi.unstubAllGlobals()

describe('store-lifetime unload guard', () => {
  it('registers one beforeunload listener for the store lifetime', () => {
    expect(typeof registered.get('beforeunload')).toBe('function')
  })

  it('prompts on reload while the store holds unsaved edits', () => {
    useDSLStore.setState({ ...initialDSLState, wasmReady: true })
    useDSLStore.getState().setDslSource('MODEL "draft" {}')

    const preventDefault = vi.fn()
    registered.get('beforeunload')!({ preventDefault, returnValue: '' })

    expect(preventDefault).toHaveBeenCalled()
  })

  it('stays silent once the edits are saved', () => {
    useDSLStore.setState({ ...initialDSLState, wasmReady: true })
    useDSLStore.getState().setDslSource('MODEL "draft" {}')
    useDSLStore.getState().loadDsl('MODEL "draft" {}')

    const preventDefault = vi.fn()
    registered.get('beforeunload')!({ preventDefault, returnValue: '' })

    expect(preventDefault).not.toHaveBeenCalled()
  })
})
