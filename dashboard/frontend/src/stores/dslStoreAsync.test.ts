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

  it('rejects an import that would replace edits made while it was loading', async () => {
    const result = deferred<DecompileResult>()
    vi.mocked(dslCompiler.decompile).mockReturnValue(result.promise)
    const request = useDSLStore.getState().importYaml('new configuration')
    useDSLStore.getState().setDslSource('unsaved edits')
    result.resolve({ dsl: 'imported DSL' })
    await expect(request).rejects.toThrow('source changed during import')
    expect(useDSLStore.getState().dslSource).toBe('unsaved edits')
  })
})
