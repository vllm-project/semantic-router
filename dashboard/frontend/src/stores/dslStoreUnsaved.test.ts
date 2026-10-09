import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { dslCompiler } from '@/lib/dslCompiler'
import { selectHasUnsavedChanges, useDSLStore } from './dslStore'
import { initialDSLState } from './dslStoreSupport'

vi.mock('@/lib/dslCompiler', () => ({
  dslCompiler: {
    init: vi.fn().mockResolvedValue(undefined),
    decompile: vi.fn(),
    format: vi.fn(),
    compile: vi.fn(),
    validate: vi.fn(),
    parseAST: vi.fn(),
  },
}))

describe('unsaved changes signal', async () => {
  beforeEach(async () => {
    await useDSLStore.getState().initCompiler()
    vi.resetAllMocks()
    useDSLStore.setState({ ...initialDSLState, compilerReady: true })
    vi.mocked(dslCompiler.validate).mockResolvedValue({ diagnostics: [], errorCount: 0 })
    vi.mocked(dslCompiler.parseAST).mockResolvedValue({ diagnostics: [], errorCount: 0 })
  })

  afterEach(() => {
    useDSLStore.getState().reset()
    vi.unstubAllGlobals()
  })

  it('keeps the unsaved signal set after compile clears staleness', async () => {
    useDSLStore.getState().setDslSource('MODEL "draft" {}')
    expect(useDSLStore.getState().dirty).toBe(true)
    expect(selectHasUnsavedChanges(useDSLStore.getState())).toBe(true)

    vi.mocked(dslCompiler.compile).mockResolvedValue({
      yaml: 'compiled output',
      diagnostics: [],
    })
    await useDSLStore.getState().compile()

    expect(useDSLStore.getState().dirty).toBe(false)
    expect(selectHasUnsavedChanges(useDSLStore.getState())).toBe(true)
  })

  it('moves the baseline on loadDsl and clears the unsaved signal', async () => {
    useDSLStore.getState().setDslSource('MODEL "draft" {}')
    useDSLStore.getState().loadDsl('MODEL "loaded" {}')

    expect(selectHasUnsavedChanges(useDSLStore.getState())).toBe(false)
    expect(useDSLStore.getState().savedSource).toBe('MODEL "loaded" {}')
  })

  it('clears the unsaved signal on reset', async () => {
    useDSLStore.getState().setDslSource('MODEL "draft" {}')
    expect(selectHasUnsavedChanges(useDSLStore.getState())).toBe(true)

    useDSLStore.getState().reset()

    expect(useDSLStore.getState().dirty).toBe(false)
    expect(selectHasUnsavedChanges(useDSLStore.getState())).toBe(false)
    expect(useDSLStore.getState().savedSource).toBe('')
  })

  it('moves the baseline on a successful deploy and leaves dirty to the compile path', async () => {
    useDSLStore.setState({
      dslSource: 'MODEL "draft" {}',
      yamlOutput: 'compiled output',
      savedSource: 'previous baseline',
      dirty: true,
    })
    vi.stubGlobal(
      'fetch',
      vi
        .fn()
        .mockImplementation(() =>
          Promise.resolve(
            new Response(JSON.stringify({ status: 'persisted', version: '3' }), { status: 200 }),
          ),
        ),
    )

    await useDSLStore.getState().executeDeploy()

    expect(useDSLStore.getState()).toMatchObject({
      deployResult: { status: 'success' },
      dirty: true,
      savedSource: 'MODEL "draft" {}',
    })
    expect(selectHasUnsavedChanges(useDSLStore.getState())).toBe(false)
  })

  it('shows how a deploy that needs a restart gets applied', async () => {
    useDSLStore.setState({
      dslSource: 'MODEL "draft" {}',
      yamlOutput: 'compiled output',
      savedSource: 'previous baseline',
      dirty: true,
    })
    const message = 'Restart required: run `vllm-sr serve` to apply.'
    vi.stubGlobal(
      'fetch',
      vi.fn().mockImplementation(() =>
        Promise.resolve(
          new Response(JSON.stringify({ status: 'restart_required', version: '5', message }), {
            status: 202,
          }),
        ),
      ),
    )

    await useDSLStore.getState().executeDeploy()

    expect(useDSLStore.getState()).toMatchObject({
      deployResult: { status: 'success', version: '5', message },
      savedSource: 'MODEL "draft" {}',
    })
  })

  it('keeps the unsaved signal when an edit lands during the deploy', async () => {
    useDSLStore.setState({
      dslSource: 'MODEL "draft" {}',
      yamlOutput: 'compiled output',
      savedSource: 'previous baseline',
      dirty: true,
    })
    vi.stubGlobal(
      'fetch',
      vi.fn().mockImplementation(() => {
        // The user keeps editing while the deploy runs, exactly as the
        // non-blocking toast progress allows.
        useDSLStore.getState().setDslSource('MODEL "draft" {} // edited while deploying')
        return Promise.resolve(
          new Response(JSON.stringify({ status: 'persisted', version: '4' }), { status: 200 }),
        )
      }),
    )

    await useDSLStore.getState().executeDeploy()

    expect(useDSLStore.getState()).toMatchObject({
      deployResult: { status: 'success' },
      savedSource: 'MODEL "draft" {}',
    })
    expect(selectHasUnsavedChanges(useDSLStore.getState())).toBe(true)
  })

  it('keeps the unsaved signal when the deploy fails', async () => {
    useDSLStore.setState({
      dslSource: 'MODEL "draft" {}',
      yamlOutput: 'compiled output',
      savedSource: 'previous baseline',
    })
    vi.stubGlobal(
      'fetch',
      vi.fn().mockImplementation(() => Promise.resolve(new Response('nope', { status: 500 }))),
    )

    await useDSLStore.getState().executeDeploy()

    expect(useDSLStore.getState()).toMatchObject({
      deployResult: { status: 'error' },
      savedSource: 'previous baseline',
    })
    expect(selectHasUnsavedChanges(useDSLStore.getState())).toBe(true)
  })
})
