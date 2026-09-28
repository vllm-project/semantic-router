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

describe('DSL operation errors', () => {
  beforeEach(async () => {
    await useDSLStore.getState().initWasm()
    vi.resetAllMocks()
    useDSLStore.setState({ ...initialDSLState, wasmReady: true })
    vi.mocked(wasmBridge.validate).mockReturnValue({ diagnostics: [], errorCount: 0 })
  })

  afterEach(() => {
    useDSLStore.getState().reset()
    vi.unstubAllGlobals()
  })

  it('preserves the decompiler error and the existing document on failed import', () => {
    const document = {
      dslSource: 'MODEL "existing" {}',
      baseConfigYaml: 'existing config',
      renderedYamlOutput: 'existing rendered config',
      yamlOutput: 'existing compiler output',
      dirty: true,
    }
    useDSLStore.setState(document)
    vi.mocked(wasmBridge.decompile).mockReturnValue({
      dsl: '',
      error: 'routing: field "unknown_field" not found',
    })

    expect(() => useDSLStore.getState().importYaml('routing: {unknown_field: true}')).toThrow(
      'routing: field "unknown_field" not found',
    )
    expect(useDSLStore.getState()).toMatchObject(document)

    vi.mocked(wasmBridge.decompile).mockReturnValue({ dsl: 'MODEL "repaired" {}' })
    useDSLStore.getState().importYaml('repaired config')
    expect(useDSLStore.getState()).toMatchObject({
      dslSource: 'MODEL "repaired" {}',
      baseConfigYaml: 'repaired config',
      diagnostics: [],
      compileError: null,
      dirty: false,
    })
  })

  it('distinguishes an unavailable compiler from invalid YAML', () => {
    useDSLStore.setState({ wasmReady: false })
    expect(() => useDSLStore.getState().importYaml('version: v0.3')).toThrow('WASM not ready')
    expect(wasmBridge.decompile).not.toHaveBeenCalled()
  })

  it('propagates router fetch and decompile failures without replacing the document', async () => {
    useDSLStore.setState({ dslSource: 'existing draft', dirty: true })
    const fetchMock = vi.fn().mockResolvedValue(new Response('unavailable', { status: 503 }))
    vi.stubGlobal('fetch', fetchMock)
    await expect(useDSLStore.getState().loadFromRouter()).rejects.toThrow('HTTP 503')
    fetchMock.mockResolvedValue(new Response('routing: {unknown_field: true}'))
    vi.mocked(wasmBridge.decompile).mockReturnValue({ dsl: '', error: 'unknown_field is invalid' })
    await expect(useDSLStore.getState().loadFromRouter()).rejects.toThrow(
      'unknown_field is invalid',
    )
    expect(useDSLStore.getState()).toMatchObject({ dslSource: 'existing draft', dirty: true })
  })

  it.each(['result', 'exception'])(
    'shows a format %s failure without editing the source',
    (kind) => {
      useDSLStore.setState({
        dslSource: 'hello',
        diagnostics: [{ level: 'warning', message: 'old', line: 1, column: 1 }],
      })
      const error = 'parse errors: [1:1: unexpected token "hello"]'
      if (kind === 'result') vi.mocked(wasmBridge.format).mockReturnValue({ dsl: '', error })
      else
        vi.mocked(wasmBridge.format).mockImplementation(() => {
          throw new Error(error)
        })

      useDSLStore.getState().format()
      expect(useDSLStore.getState()).toMatchObject({
        dslSource: 'hello',
        compileError: error,
        diagnostics: [],
        dirty: false,
      })

      vi.mocked(wasmBridge.format).mockReturnValue({ dsl: 'MODEL "repaired" {}\n' })
      useDSLStore.getState().format()
      expect(useDSLStore.getState()).toMatchObject({
        dslSource: 'MODEL "repaired" {}\n',
        compileError: null,
        dirty: true,
      })
    },
  )

  it.each(['validate', 'parseAST'] as const)(
    'replaces stale diagnostics when %s throws',
    (action) => {
      useDSLStore.setState({
        dslSource: 'MODEL "draft" {}',
        diagnostics: [{ level: 'warning', message: 'old', line: 1, column: 1 }],
      })
      vi.mocked(wasmBridge[action]).mockImplementation(() => {
        throw new Error('Compiler unavailable')
      })
      useDSLStore.getState()[action]()
      expect(useDSLStore.getState()).toMatchObject({
        dslSource: 'MODEL "draft" {}',
        diagnostics: [],
        compileError: 'Compiler unavailable',
      })
      vi.mocked(wasmBridge[action]).mockReturnValue({ diagnostics: [], errorCount: 0 })
      useDSLStore.getState()[action]()
      expect(useDSLStore.getState().compileError).toBeNull()
    },
  )

  it('clears diagnostics for the previous text as the user edits', () => {
    useDSLStore.setState({
      compileError: 'previous error',
      diagnostics: [{ level: 'error', message: 'old', line: 1, column: 1 }],
    })
    useDSLStore.getState().setDslSource('MODEL "repaired" {}')
    expect(useDSLStore.getState()).toMatchObject({
      compileError: null,
      diagnostics: [],
      dirty: true,
    })
  })

  it('does not preview or deploy stale output after a compile exception', () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response('{}'))
    vi.stubGlobal('fetch', fetchMock)
    useDSLStore.setState({ dslSource: 'hello', yamlOutput: 'old compiled config', dirty: true })
    vi.mocked(wasmBridge.compile).mockImplementation(() => {
      throw new Error('Compiler unavailable')
    })
    useDSLStore.getState().requestDeploy()
    expect(useDSLStore.getState()).toMatchObject({
      compileError: 'Compiler unavailable',
      showDeployConfirm: false,
      deployResult: { status: 'error' },
    })
    expect(fetchMock).not.toHaveBeenCalled()
  })

  it('refuses compiler output accompanied by an error even without diagnostic entries', () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response('{}'))
    vi.stubGlobal('fetch', fetchMock)
    useDSLStore.setState({ dslSource: 'MODEL "draft" {}', dirty: true })
    vi.mocked(wasmBridge.compile).mockReturnValue({
      yaml: 'partial output',
      diagnostics: [],
      error: 'Compilation failed',
    })
    useDSLStore.getState().requestDeploy()
    expect(useDSLStore.getState()).toMatchObject({
      showDeployConfirm: false,
      deployResult: { status: 'error' },
    })
    expect(fetchMock).not.toHaveBeenCalled()
  })
})
