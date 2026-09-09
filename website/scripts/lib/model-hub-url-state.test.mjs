import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { dirname, resolve } from 'node:path'
import { test } from 'node:test'
import { fileURLToPath } from 'node:url'
import ts from 'typescript'

const repositoryRoot = resolve(dirname(fileURLToPath(import.meta.url)), '../../..')
const directoryHelperPath = resolve(
  repositoryRoot,
  'website/src/data/modelHubDirectorySupport.ts',
)
const urlHelperPath = resolve(repositoryRoot, 'website/src/data/modelHubUrlState.ts')
const directoryHelperUrl = `data:text/javascript;base64,${Buffer.from(
  ts.transpileModule(readFileSync(directoryHelperPath, 'utf8'), {
    compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2020 },
    fileName: directoryHelperPath,
  }).outputText,
).toString('base64')}`
const urlHelperJavaScript = ts
  .transpileModule(readFileSync(urlHelperPath, 'utf8'), {
    compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2020 },
    fileName: urlHelperPath,
  })
  .outputText.replace('./modelHubDirectorySupport', directoryHelperUrl)
const support = await import(
  `data:text/javascript;base64,${Buffer.from(urlHelperJavaScript).toString('base64')}`,
)

test('model hub URL state defaults are compact and stable', () => {
  const state = support.parseModelHubUrlState('')

  assert.deepEqual(state, {
    filters: {
      search: '',
      kind: 'all',
      distribution: 'all',
      publisher: 'all',
      provider: 'all',
      capability: 'all',
      lifecycle: 'supported',
      sort: 'newest',
    },
    view: 'list',
    page: 1,
    benchmark: { filter: 'all', query: '', publisher: 'all' },
    arenaScope: 'all',
    arenaLayer: 'overall',
    arenaCapability: '',
    arenaBenchmark: '',
    selectedModelID: null,
  })
  assert.equal(support.serializeModelHubUrlState(state), '')
})

test('every model hub control round-trips through the URL', () => {
  const search = '?q=mixture+router&kind=virtual&distribution=router_recipe'
    + '&creator=vLLM&provider=openai&capability=tools&lifecycle=all&sort=context'
    + '&view=table&page=3&benchmark=tag%3Acore&benchmark_q=astra'
    + '&benchmark_creator=OpenAI&arena=virtual&arena_layer=benchmarks'
    + '&arena_capability=vllm-sr%2Fcoding%401.0.0'
    + '&arena_benchmark=livecodebench%2Flivecodebench%406.0.0'
    + '&model=openai%2Fgpt-6-astra'
  const state = support.parseModelHubUrlState(search)

  assert.deepEqual(state.filters, {
    search: 'mixture router',
    kind: 'virtual',
    distribution: 'router_recipe',
    publisher: 'vLLM',
    provider: 'openai',
    capability: 'tools',
    lifecycle: 'all',
    sort: 'context',
  })
  assert.equal(state.view, 'table')
  assert.equal(state.page, 3)
  assert.deepEqual(state.benchmark, {
    filter: 'tag:core',
    query: 'astra',
    publisher: 'OpenAI',
  })
  assert.equal(state.arenaScope, 'virtual')
  assert.equal(state.arenaLayer, 'benchmarks')
  assert.equal(state.arenaCapability, 'vllm-sr/coding@1.0.0')
  assert.equal(state.arenaBenchmark, 'livecodebench/livecodebench@6.0.0')
  assert.equal(state.selectedModelID, 'openai/gpt-6-astra')
  assert.deepEqual(
    support.parseModelHubUrlState(support.serializeModelHubUrlState(state)),
    state,
  )
})

test('invalid enum and page values fall back safely', () => {
  const state = support.parseModelHubUrlState(
    '?kind=mixture&distribution=download&lifecycle=retired&sort=rank&view=cards&page=-2&arena=closed&arena_layer=domains',
  )

  assert.equal(state.filters.kind, 'all')
  assert.equal(state.filters.distribution, 'all')
  assert.equal(state.filters.lifecycle, 'supported')
  assert.equal(state.filters.sort, 'newest')
  assert.equal(state.view, 'list')
  assert.equal(state.page, 1)
  assert.equal(state.arenaScope, 'all')
  assert.equal(state.arenaLayer, 'overall')
})

test('serialization preserves unrelated campaign parameters', () => {
  const state = support.parseModelHubUrlState('?utm_source=weekly')
  state.filters.kind = 'virtual'
  state.selectedModelID = 'virtual/team/router'

  const search = support.serializeModelHubUrlState(state, '?utm_source=weekly&kind=physical')
  const parameters = new URLSearchParams(search)
  assert.equal(parameters.get('utm_source'), 'weekly')
  assert.equal(parameters.get('kind'), 'virtual')
  assert.equal(parameters.get('model'), 'virtual/team/router')
})
