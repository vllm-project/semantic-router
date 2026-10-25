import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { dslCompiler } from '@/lib/dslCompiler'
import { useDSLStore } from './dslStore'
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

describe('DSL operation errors', async () => {
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

  it('preserves the decompiler error and the existing document on failed import', async () => {
    const document = {
      dslSource: 'MODEL "existing" {}',
      baseConfigYaml: 'existing config',
      renderedYamlOutput: 'existing rendered config',
      yamlOutput: 'existing compiler output',
      dirty: true,
    }
    useDSLStore.setState(document)
    vi.mocked(dslCompiler.decompile).mockResolvedValue({
      dsl: '',
      error: 'routing: field "unknown_field" not found',
    })

    await expect(useDSLStore.getState().importYaml('routing: {unknown_field: true}')).rejects.toThrow(
      'routing: field "unknown_field" not found',
    )
    expect(useDSLStore.getState()).toMatchObject(document)

    vi.mocked(dslCompiler.decompile).mockResolvedValue({ dsl: 'MODEL "repaired" {}' })
    await useDSLStore.getState().importYaml('repaired config')
    expect(useDSLStore.getState()).toMatchObject({
      dslSource: 'MODEL "repaired" {}',
      baseConfigYaml: 'repaired config',
      diagnostics: [],
      compileError: null,
      dirty: false,
    })
  })

  it('distinguishes an unavailable compiler from invalid YAML', async () => {
    useDSLStore.setState({ compilerReady: false })
    await expect(useDSLStore.getState().importYaml('version: v0.3')).rejects.toThrow('Compiler not ready')
    expect(dslCompiler.decompile).not.toHaveBeenCalled()
  })

  it('propagates router fetch and decompile failures without replacing the document', async () => {
    useDSLStore.setState({ dslSource: 'existing draft', dirty: true })
    const fetchMock = vi.fn().mockResolvedValue(new Response('unavailable', { status: 503 }))
    vi.stubGlobal('fetch', fetchMock)
    await expect(useDSLStore.getState().loadFromRouter()).rejects.toThrow('HTTP 503')
    fetchMock.mockResolvedValue(new Response('routing: {unknown_field: true}'))
    vi.mocked(dslCompiler.decompile).mockResolvedValue({ dsl: '', error: 'unknown_field is invalid' })
    await expect(useDSLStore.getState().loadFromRouter()).rejects.toThrow(
      'unknown_field is invalid',
    )
    expect(useDSLStore.getState()).toMatchObject({ dslSource: 'existing draft', dirty: true })
  })

  it.each(['result', 'exception'])(
    'shows a format %s failure without editing the source',
    async (kind) => {
      useDSLStore.setState({
        dslSource: 'hello',
        diagnostics: [{ level: 'warning', message: 'old', line: 1, column: 1 }],
      })
      const error = 'parse errors: [1:1: unexpected token "hello"]'
      if (kind === 'result') vi.mocked(dslCompiler.format).mockResolvedValue({ dsl: '', error })
      else
        vi.mocked(dslCompiler.format).mockImplementation(() => {
          throw new Error(error)
        })

      await useDSLStore.getState().format()
      expect(useDSLStore.getState()).toMatchObject({
        dslSource: 'hello',
        compileError: error,
        diagnostics: [],
        dirty: false,
      })

      vi.mocked(dslCompiler.format).mockResolvedValue({ dsl: 'MODEL "repaired" {}\n' })
      await useDSLStore.getState().format()
      expect(useDSLStore.getState()).toMatchObject({
        dslSource: 'MODEL "repaired" {}\n',
        compileError: null,
        dirty: true,
      })
    },
  )

  it.each(['validate', 'parseAST'] as const)(
    'replaces stale diagnostics when %s throws',
    async (action) => {
      useDSLStore.setState({
        dslSource: 'MODEL "draft" {}',
        diagnostics: [{ level: 'warning', message: 'old', line: 1, column: 1 }],
      })
      vi.mocked(dslCompiler[action]).mockImplementation(() => {
        throw new Error('Compiler unavailable')
      })
      await useDSLStore.getState()[action]()
      expect(useDSLStore.getState()).toMatchObject({
        dslSource: 'MODEL "draft" {}',
        diagnostics: [],
        compileError: 'Compiler unavailable',
      })
      vi.mocked(dslCompiler[action]).mockResolvedValue({ diagnostics: [], errorCount: 0 })
      await useDSLStore.getState()[action]()
      expect(useDSLStore.getState().compileError).toBeNull()
    },
  )

  it('clears diagnostics for the previous text as the user edits', async () => {
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

  it('does not preview or deploy stale output after a compile exception', async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response('{}'))
    vi.stubGlobal('fetch', fetchMock)
    useDSLStore.setState({ dslSource: 'hello', yamlOutput: 'old compiled config', dirty: true })
    vi.mocked(dslCompiler.compile).mockImplementation(() => {
      throw new Error('Compiler unavailable')
    })
    await useDSLStore.getState().requestDeploy()
    expect(useDSLStore.getState()).toMatchObject({
      compileError: 'Compiler unavailable',
      showDeployConfirm: false,
      deployResult: { status: 'error' },
    })
    expect(fetchMock).not.toHaveBeenCalled()
  })

  it('refuses compiler output accompanied by an error even without diagnostic entries', async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response('{}'))
    vi.stubGlobal('fetch', fetchMock)
    useDSLStore.setState({ dslSource: 'MODEL "draft" {}', dirty: true })
    vi.mocked(dslCompiler.compile).mockResolvedValue({
      yaml: 'partial output',
      diagnostics: [],
      error: 'Compilation failed',
    })
    await useDSLStore.getState().requestDeploy()
    expect(useDSLStore.getState()).toMatchObject({
      showDeployConfirm: false,
      deployResult: { status: 'error' },
    })
    expect(fetchMock).not.toHaveBeenCalled()
  })
})
