import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { dirname, resolve } from 'node:path'
import { test } from 'node:test'
import { fileURLToPath } from 'node:url'
import ts from 'typescript'

const repositoryRoot = resolve(dirname(fileURLToPath(import.meta.url)), '../../..')
const helperPath = resolve(repositoryRoot, 'website/src/data/modelHubArenaSupport.ts')
const helperJavaScript = ts.transpileModule(readFileSync(helperPath, 'utf8'), {
  compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2020 },
  fileName: helperPath,
}).outputText
const support = await import(
  `data:text/javascript;base64,${Buffer.from(helperJavaScript).toString('base64')}`,
)

const component = (benchmark, status) => ({
  benchmark,
  metric: 'accuracy',
  benchmark_profile: 'standard',
  weight: 0.5,
  status,
})

const catalog = {
  catalogs: [{ default_intelligence_index: 'test/index@1.0.0' }],
  indices: [{
    id: 'test/index@1.0.0',
    display_name: 'Test Index',
    components: [],
  }],
  benchmarks: [
    { id: 'test/a@1.0.0', display_name: 'A' },
    { id: 'test/b@1.0.0', display_name: 'B' },
  ],
  reasoning_families: [{
    id: 'reasoning/test@1.0.0',
    levels: ['low', 'high'],
  }],
  models: [
    {
      id: 'creator/open',
      display_name: 'Open',
      kind: 'physical',
      publisher: 'Creator',
      distribution: { type: 'open_weights' },
      reasoning_family: 'reasoning/test@1.0.0',
    },
    {
      id: 'creator/closed',
      display_name: 'Closed',
      kind: 'physical',
      publisher: 'Creator',
      distribution: { type: 'proprietary_api' },
    },
    {
      id: 'virtual/router',
      display_name: 'Router',
      kind: 'virtual',
      publisher: 'vLLM',
      distribution: { type: 'router_recipe' },
    },
  ],
  index_results: [
    {
      model: 'creator/open',
      reasoning_effort: 'low',
      index: 'test/index@1.0.0',
      status: 'available',
      score: 80,
      coverage: 1,
      components: [component('test/a@1.0.0', 'available')],
    },
    {
      model: 'creator/open',
      reasoning_effort: 'high',
      index: 'test/index@1.0.0',
      status: 'available',
      score: 75,
      coverage: 1,
      components: [component('test/a@1.0.0', 'available')],
    },
    {
      model: 'creator/closed',
      reasoning_effort: 'default',
      index: 'test/index@1.0.0',
      status: 'partial',
      score: null,
      coverage: 0.5,
      components: [
        component('test/a@1.0.0', 'available'),
        component('test/b@1.0.0', 'missing'),
      ],
    },
    {
      model: 'virtual/router',
      reasoning_effort: 'default',
      index: 'test/index@1.0.0',
      status: 'available',
      score: 90,
      coverage: 1,
      components: [component('test/a@1.0.0', 'available')],
    },
  ],
}

test('arena ranks physical and virtual models with the same complete-case rule', () => {
  const arena = support.modelHubArenaData(catalog, 'all')

  assert.deepEqual(
    arena.ranked.map(row => [row.rank, row.model.id, row.result.reasoning_effort]),
    [[1, 'virtual/router', 'default'], [2, 'creator/open', 'high']],
  )
  assert.deepEqual(arena.awaitingEvidence.map(row => row.model.id), ['creator/closed'])
  assert.deepEqual(arena.awaitingEvidence[0].missingBenchmarks, ['B'])
})

test('arena scopes are shareable projections over the same index results', () => {
  assert.deepEqual(
    support.modelHubArenaData(catalog, 'open').ranked.map(row => row.model.id),
    ['creator/open'],
  )
  assert.deepEqual(
    support.modelHubArenaData(catalog, 'virtual').ranked.map(row => row.model.id),
    ['virtual/router'],
  )
})
