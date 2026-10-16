import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { wasmBridge } from '@/lib/wasm'
import { useDSLStore } from './dslStore'
import { initialDSLState } from './dslStoreSupport'

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

// The preview endpoint reports the Router's verdict on the merged config
// beside the diff. The store has to carry it separately from a failed fetch:
// a refused document still has a diff worth showing, and only the deploy
// button should be blocked.
describe('deploy preview validation verdict', () => {
  beforeEach(async () => {
    await useDSLStore.getState().initWasm()
    vi.resetAllMocks()
    useDSLStore.setState({
      ...initialDSLState,
      wasmReady: true,
      dslSource: 'SIGNAL complexity r { threshold: 0.1 hard_above: 0.85 easy_below: 0.6 }',
      yamlOutput: 'routing: {}',
      dirty: false,
    })
    vi.mocked(wasmBridge.validate).mockReturnValue({ diagnostics: [], errorCount: 0 })
  })

  afterEach(() => {
    useDSLStore.getState().reset()
    vi.unstubAllGlobals()
  })

  function stubPreview(body: Record<string, unknown>) {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue({ ok: true, json: async () => body }),
    )
  }

  it('keeps the diff and records the verdict when the Router refuses the merged config', async () => {
    stubPreview({
      current: 'routing: {}',
      preview: 'routing:\n  signals: {}',
      validation_error: 'Merged config validation failed: keep one',
    })

    useDSLStore.getState().requestDeploy()
    await vi.waitFor(() => expect(useDSLStore.getState().deployPreviewLoading).toBe(false))

    expect(useDSLStore.getState()).toMatchObject({
      showDeployConfirm: true,
      deployPreviewError: null,
      deployPreviewMerged: 'routing:\n  signals: {}',
      deployPreviewValidationError: 'Merged config validation failed: keep one',
    })
  })

  it('clears a previous verdict when the next preview loads cleanly', async () => {
    useDSLStore.setState({ deployPreviewValidationError: 'stale verdict' })
    stubPreview({ current: 'routing: {}', preview: 'routing: {}' })

    useDSLStore.getState().requestDeploy()
    await vi.waitFor(() => expect(useDSLStore.getState().deployPreviewLoading).toBe(false))

    expect(useDSLStore.getState().deployPreviewValidationError).toBeNull()
  })
})
