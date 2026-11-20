import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { useDSLStore } from './dslStore'
import { initialDSLState } from './dslStoreSupport'

vi.mock('@/lib/dslCompiler', () => ({
  dslCompiler: {
    init: vi.fn(),
    decompile: vi.fn(),
    compile: vi.fn(),
    validate: vi.fn(),
    parseAST: vi.fn(),
  },
}))

const okPreview = (current: string, preview: string) =>
  new Response(JSON.stringify({ current, preview }))

const failedPreview = () =>
  new Response(JSON.stringify({ message: 'preview failed' }), { status: 500 })

// Each deploy-preview request stays pending until the test resolves it, so a
// stale response can land while a newer preview is still in flight.
function stubDeferredPreviews() {
  const pending: { resolve: (response: Response) => void }[] = []
  const fetchMock = vi.fn(async (url: string) => {
    if (url !== '/api/router/config/deploy/preview') return new Response(JSON.stringify({}))
    let resolve!: (response: Response) => void
    const promise = new Promise<Response>((res) => {
      resolve = res
    })
    pending.push({ resolve })
    return promise
  })
  vi.stubGlobal('fetch', fetchMock)
  return {
    pending,
    previewBodies: () =>
      fetchMock.mock.calls
        .filter(([url]) => url === '/api/router/config/deploy/preview')
        .map(([, init]) => JSON.parse(String(init.body)) as { yaml: string }),
  }
}

const flush = () => new Promise((resolve) => setTimeout(resolve, 0))

function openPreview(yaml: string, dsl: string) {
  useDSLStore.setState({ yamlOutput: yaml, dslSource: dsl, dirty: false })
  void useDSLStore.getState().requestDeploy()
}

describe('Builder deploy preview staleness', () => {
  beforeEach(() => {
    vi.resetAllMocks()
    useDSLStore.setState({ ...initialDSLState, compilerReady: true })
  })

  afterEach(() => {
    useDSLStore.getState().reset()
    vi.unstubAllGlobals()
  })

  // A response from a cancelled preview must not fill the dialog for the next
  // draft: the operator would confirm one diff while Deploy submits another.
  it('ignores a cancelled preview response over the next draft', async () => {
    const { pending, previewBodies } = stubDeferredPreviews()
    openPreview('recipes: [a]\n', 'ROUTE a {}')
    useDSLStore.getState().dismissDeploy()
    openPreview('recipes: [b]\n', 'ROUTE b {}')

    pending[0].resolve(okPreview('current a', 'preview a'))
    await flush()

    let state = useDSLStore.getState()
    expect(state.showDeployConfirm).toBe(true)
    expect(state.deployPreviewLoading).toBe(true)
    expect(state.deployPreviewCurrent).toBe('')
    expect(state.deployPreviewMerged).toBe('')

    pending[1].resolve(okPreview('current b', 'preview b'))
    await vi.waitFor(() => expect(useDSLStore.getState().deployPreviewLoading).toBe(false))
    state = useDSLStore.getState()
    expect(state.deployPreviewCurrent).toBe('current b')
    expect(state.deployPreviewMerged).toBe('preview b')
    expect(previewBodies()[1]).toMatchObject({ yaml: 'recipes: [b]\n' })
  })

  it('leaves the dismissed dialog untouched when its preview responds late', async () => {
    const { pending } = stubDeferredPreviews()
    openPreview('recipes: []\n', 'ROUTE a {}')
    useDSLStore.getState().dismissDeploy()

    pending[0].resolve(okPreview('current a', 'preview a'))
    await flush()

    const state = useDSLStore.getState()
    expect(state.showDeployConfirm).toBe(false)
    expect(state.deployPreviewLoading).toBe(false)
    expect(state.deployPreviewCurrent).toBe('')
    expect(state.deployPreviewMerged).toBe('')
    expect(state.deployPreviewError).toBeNull()
  })

  it('ignores a stale preview error while a newer preview is pending', async () => {
    const { pending } = stubDeferredPreviews()
    openPreview('recipes: [a]\n', 'ROUTE a {}')
    useDSLStore.getState().dismissDeploy()
    openPreview('recipes: [b]\n', 'ROUTE b {}')

    pending[0].resolve(failedPreview())
    await flush()

    const state = useDSLStore.getState()
    expect(state.deployPreviewLoading).toBe(true)
    expect(state.deployPreviewError).toBeNull()
  })
})
