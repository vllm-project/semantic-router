import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { dslCompiler } from '@/lib/dslCompiler'
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

const loadedConfig = 'version: v0.5\nrecipes:\n  - name: default\n'

function stubRouter() {
  const fetchMock = vi.fn(async (url: string) => {
    if (url === '/api/router/config/yaml') return new Response(loadedConfig)
    if (url === '/api/router/config/deploy') {
      return new Response(JSON.stringify({ status: 'persisted', version: 'v2' }))
    }
    return new Response(JSON.stringify({ current: '', preview: 'rendered over the live config' }))
  })
  vi.stubGlobal('fetch', fetchMock)
  return (url: string) => {
    const call = fetchMock.mock.calls.find(([called]) => called === url) as unknown as
      | [string, RequestInit]
      | undefined
    return JSON.parse(String(call?.[1].body)) as { baseYaml: string; mode: string }
  }
}

async function previewAndDeploy() {
  useDSLStore.setState({ yamlOutput: 'recipes: []\n', dirty: false })
  await useDSLStore.getState().requestDeploy()
  await useDSLStore.getState().executeDeploy()
}

describe('Builder deploy base', () => {
  beforeEach(() => {
    vi.resetAllMocks()
    useDSLStore.setState({ ...initialDSLState, compilerReady: true })
    vi.mocked(dslCompiler.decompile).mockResolvedValue({ dsl: 'ROUTE default {}' })
    vi.mocked(dslCompiler.validate).mockResolvedValue({ diagnostics: [], errorCount: 0 })
    vi.mocked(dslCompiler.parseAST).mockResolvedValue({ diagnostics: [], errorCount: 0 })
  })

  afterEach(() => {
    useDSLStore.getState().reset()
    vi.unstubAllGlobals()
  })

  // A copy loaded from the Router goes stale when another page saves, so the
  // deploy applies the draft to the config the Router has at deploy time.
  it('applies a Router-loaded draft to the live config', async () => {
    const body = stubRouter()
    await useDSLStore.getState().loadFromRouter()
    await previewAndDeploy()

    expect(body('/api/router/config/deploy/preview')).toMatchObject({
      baseYaml: '',
      mode: 'replace',
    })
    expect(body('/api/router/config/deploy')).toMatchObject({ baseYaml: '', mode: 'replace' })
  })

  it('renders the YAML preview of a Router-loaded draft over the live config', async () => {
    const body = stubRouter()
    vi.mocked(dslCompiler.compile).mockResolvedValue({ yaml: 'recipes: []\n', diagnostics: [] })
    await useDSLStore.getState().loadFromRouter()
    await useDSLStore.getState().compile()

    await vi.waitFor(() =>
      expect(useDSLStore.getState().renderedYamlOutput).toBe('rendered over the live config'),
    )
    expect(body('/api/router/config/deploy/preview')).toMatchObject({
      baseYaml: '',
      mode: 'replace',
    })
  })

  it('still deploys an imported config as the new base', async () => {
    const body = stubRouter()
    await useDSLStore.getState().loadFromRouter()
    await useDSLStore.getState().importYaml('version: v0.5\nlisteners: []\n')
    await previewAndDeploy()

    expect(body('/api/router/config/deploy')).toMatchObject({
      baseYaml: 'version: v0.5\nlisteners: []\n',
      mode: 'replace',
    })
  })
})
