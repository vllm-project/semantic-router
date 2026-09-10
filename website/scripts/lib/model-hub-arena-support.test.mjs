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

const indexResult = (model, index, effort, status, score) => ({
  model,
  reasoning_effort: effort,
  index,
  status,
  score,
  coverage: status === 'available' ? 1 : 0.5,
  components: [],
})

const catalog = {
  catalogs: [{ default_intelligence_index: 'test/intelligence@1.0.0' }],
  indices: [
    {
      id: 'test/intelligence@1.0.0',
      display_name: 'Test Intelligence',
      description: 'Overall',
      components: [{ index: 'test/general@1.0.0', weight: 1 }],
    },
    {
      id: 'test/general@1.0.0',
      display_name: 'General',
      description: 'General capability',
      components: [{
        benchmark: 'test/a@1.0.0',
        metric: 'accuracy',
        benchmark_profiles: ['independent', 'published'],
        weight: 1,
        normalization: { type: 'identity' },
      }],
    },
  ],
  benchmarks: [{
    id: 'test/a@1.0.0',
    display_name: 'A',
    default_profile: 'published',
    metrics: [{ id: 'accuracy', unit: 'proportion', range: [0, 1] }],
  }],
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
      id: 'creator/partial',
      display_name: 'Partial',
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
  evaluations: [
    {
      id: 'eval/open-low', model: 'creator/open', benchmark: 'test/a@1.0.0',
      benchmark_profile: 'published', reasoning_effort: 'low', status: 'available',
      metrics: { accuracy: 0.7 },
    },
    {
      id: 'eval/open-high', model: 'creator/open', benchmark: 'test/a@1.0.0',
      benchmark_profile: 'independent', reasoning_effort: 'high', status: 'available',
      metrics: { accuracy: 0.8 },
    },
    {
      id: 'eval/partial', model: 'creator/partial', benchmark: 'test/a@1.0.0',
      benchmark_profile: 'published', reasoning_effort: 'default', status: 'available',
      metrics: { accuracy: 0.95 },
    },
    {
      id: 'eval/router', model: 'virtual/router', benchmark: 'test/a@1.0.0',
      benchmark_profile: 'published', reasoning_effort: 'default', status: 'available',
      metrics: { accuracy: 0.9 },
    },
  ],
  index_results: [
    indexResult('creator/open', 'test/intelligence@1.0.0', 'low', 'available', 80),
    indexResult('creator/open', 'test/intelligence@1.0.0', 'high', 'available', 75),
    indexResult('creator/partial', 'test/intelligence@1.0.0', 'default', 'partial', null),
    indexResult('virtual/router', 'test/intelligence@1.0.0', 'default', 'available', 90),
    indexResult('creator/open', 'test/general@1.0.0', 'high', 'available', 80),
    indexResult('creator/partial', 'test/general@1.0.0', 'default', 'available', 95),
    indexResult('virtual/router', 'test/general@1.0.0', 'default', 'available', 90),
  ],
}

test('arena builds overall, capability, and benchmark ranks from one hierarchy', () => {
  const arena = support.modelHubArenaData(catalog, 'all')

  assert.deepEqual(
    arena.overall.rows.map(row => [row.rank, row.model.id, row.reasoningEffort]),
    [[1, 'virtual/router', 'default'], [2, 'creator/open', 'high']],
  )
  assert.deepEqual(
    arena.capabilities[0].rows.map(row => row.model.id),
    ['creator/partial', 'virtual/router', 'creator/open'],
  )
  assert.deepEqual(
    arena.benchmarks[0].rows.map(row => [row.model.id, row.score, row.reasoningEffort]),
    [
      ['creator/partial', 95, 'default'],
      ['virtual/router', 90, 'default'],
      ['creator/open', 80, 'high'],
    ],
  )
})

test('arena scopes physical and virtual models identically at every layer', () => {
  const open = support.modelHubArenaData(catalog, 'open')
  const virtual = support.modelHubArenaData(catalog, 'virtual')

  assert.deepEqual(open.overall.rows.map(row => row.model.id), ['creator/open'])
  assert.deepEqual(open.benchmarks[0].rows.map(row => row.model.id), ['creator/open'])
  assert.deepEqual(virtual.overall.rows.map(row => row.model.id), ['virtual/router'])
  assert.deepEqual(virtual.capabilities[0].rows.map(row => row.model.id), ['virtual/router'])
})
