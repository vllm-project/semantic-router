import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { dslCompiler } from '@/lib/dslCompiler'
import type { CompileResult, DecompileResult, FormatResult, ValidateResult } from '@/types/dsl'
import { useDSLStore } from './dslStore'
import { initialDSLState } from './dslStoreSupport'

vi.mock('@/lib/dslCompiler', () => ({
  dslCompiler: {
    init: vi.fn(),
    compile: vi.fn(),
    validate: vi.fn(),
    parseAST: vi.fn(),
    decompile: vi.fn(),
    format: vi.fn(),
    cancelPending: vi.fn(),
  },
}))

function deferred<T>() {
  let resolve!: (value: T) => void
  const promise = new Promise<T>((done) => {
    resolve = done
  })
  return { promise, resolve }
}

describe('asynchronous compiler ordering', () => {
  beforeEach(() => {
    vi.resetAllMocks()
    useDSLStore.setState({ ...initialDSLState, compilerReady: true, dslSource: 'draft A' })
  })
  afterEach(() => useDSLStore.getState().reset())

  it('keeps enclosing rule limits on imported drafts in every editor mode', async () => {
    const baseYaml = 'global:\n  router:\n    decision_rule_limits:\n      max_depth: 32\n'
    vi.mocked(dslCompiler.decompile).mockResolvedValue({ dsl: 'imported DSL' })
    vi.mocked(dslCompiler.validate).mockResolvedValue({ diagnostics: [], errorCount: 0 })
    vi.mocked(dslCompiler.parseAST).mockResolvedValue({ diagnostics: [], errorCount: 0 })
    vi.mocked(dslCompiler.compile).mockResolvedValue({ yaml: '', diagnostics: [] })
    vi.mocked(dslCompiler.format).mockResolvedValue({ dsl: 'imported DSL' })

    await useDSLStore.getState().importYaml(baseYaml)
    await useDSLStore.getState().validate()
    await useDSLStore.getState().parseAST()
    await useDSLStore.getState().compile()
    await useDSLStore.getState().format()

    expect(dslCompiler.validate).toHaveBeenLastCalledWith('imported DSL', baseYaml)
    expect(dslCompiler.parseAST).toHaveBeenLastCalledWith('imported DSL', baseYaml)
    expect(dslCompiler.compile).toHaveBeenLastCalledWith('imported DSL', baseYaml)
    expect(dslCompiler.format).toHaveBeenLastCalledWith('imported DSL', baseYaml)
  })

  it('does not publish or preview an old compilation after an A → B → A edit', async () => {
    const result = deferred<CompileResult>()
    vi.mocked(dslCompiler.compile).mockReturnValue(result.promise)
    const request = useDSLStore.getState().requestDeploy()
    useDSLStore.getState().setDslSource('draft B')
    useDSLStore.getState().setDslSource('draft A')
    result.resolve({ yaml: 'outdated configuration', diagnostics: [] })
    await request
    expect(useDSLStore.getState()).toMatchObject({
      yamlOutput: '',
      dirty: true,
      loading: false,
      showDeployConfirm: false,
    })
  })

  it('retains the newest diagnostics when requests finish out of order', async () => {
    const older = deferred<ValidateResult>()
    vi.mocked(dslCompiler.validate)
      .mockReturnValueOnce(older.promise)
      .mockResolvedValueOnce({ diagnostics: [], errorCount: 0 })
    const request = useDSLStore.getState().validate()
    await useDSLStore.getState().validate()
    older.resolve({ diagnostics: [], errorCount: 1, error: 'obsolete error' })
    await request
    expect(useDSLStore.getState().compileError).toBeNull()
  })

  it('does not let a slow format overwrite a new draft', async () => {
    const result = deferred<FormatResult>()
    vi.mocked(dslCompiler.format).mockReturnValue(result.promise)
    const request = useDSLStore.getState().format()
    useDSLStore.getState().setDslSource('new draft')
    result.resolve({ dsl: 'formatted old draft' })
    await request
    expect(useDSLStore.getState().dslSource).toBe('new draft')
  })

  it('preserves explicit format errors over pending or older validation', async () => {
    vi.useFakeTimers()
    try {
      const older = deferred<ValidateResult>()
      vi.mocked(dslCompiler.validate).mockReturnValue(older.promise)
      vi.mocked(dslCompiler.format).mockResolvedValue({
        dsl: '',
        error: 'parse errors: unexpected token "hello"',
      })
      useDSLStore.getState().setDslSource('hello')
      const validation = useDSLStore.getState().validate()
      await useDSLStore.getState().format()
      await vi.runAllTimersAsync()
      expect(dslCompiler.validate).toHaveBeenCalledOnce()
      older.resolve({ diagnostics: [], errorCount: 1, error: 'older validation error' })
      await validation
      expect(useDSLStore.getState().compileError).toBe('parse errors: unexpected token "hello"')
    } finally {
      vi.useRealTimers()
    }
  })

  it('rejects an import that would replace edits made while it was loading', async () => {
    const result = deferred<DecompileResult>()
    vi.mocked(dslCompiler.decompile).mockReturnValue(result.promise)
    const request = useDSLStore.getState().importYaml('new configuration')
    useDSLStore.getState().setDslSource('unsaved edits')
    result.resolve({ dsl: 'imported DSL' })
    await expect(request).rejects.toThrow('source changed during import')
    expect(useDSLStore.getState().dslSource).toBe('unsaved edits')
  })

  it('cancels page work on departure without discarding edits or launching a late parse', async () => {
    const result = deferred<DecompileResult>()
    vi.mocked(dslCompiler.decompile).mockReturnValue(result.promise)
    const request = useDSLStore.getState().importYaml('configuration')
    useDSLStore.setState({ dslSource: 'unsaved draft', dirty: true })
    useDSLStore.getState().pauseEditorWork()
    result.resolve({ dsl: 'late import' })
    await expect(request).rejects.toThrow('source changed during import')
    expect(dslCompiler.cancelPending).toHaveBeenCalledOnce()
    expect(dslCompiler.parseAST).not.toHaveBeenCalled()
    expect(useDSLStore.getState()).toMatchObject({ dslSource: 'unsaved draft', dirty: true })
  })
})
