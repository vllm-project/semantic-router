import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { dirname, resolve } from 'node:path'
import { test } from 'node:test'
import { fileURLToPath } from 'node:url'
import ts from 'typescript'

const repositoryRoot = resolve(
  dirname(fileURLToPath(import.meta.url)),
  '../../..',
)
const helperPath = resolve(
  repositoryRoot,
  'website/src/data/modelHubBenchmarkSupport.ts',
)
const helperJavaScript = ts.transpileModule(readFileSync(helperPath, 'utf8'), {
  compilerOptions: {
    module: ts.ModuleKind.ESNext,
    target: ts.ScriptTarget.ES2020,
  },
  fileName: helperPath,
}).outputText
const support = await import(
  `data:text/javascript;base64,${Buffer.from(helperJavaScript).toString('base64')}`,
)

test('website benchmark controls expose only available exact tuples', () => {
  const counts = support.modelHubBenchmarkSelectionCounts([
    {
      status: 'available',
      benchmark: 'bench@1',
      benchmark_profile: 'agent',
      metrics: { score: 0.8, lower_bound: null },
    },
    {
      status: 'available',
      benchmark: 'bench@1',
      benchmark_profile: 'independent',
      metrics: { score: 0.9, lower_bound: 0.85 },
    },
    {
      status: 'missing',
      benchmark: 'bench@1',
      benchmark_profile: 'unused',
      metrics: { score: 0 },
    },
  ])

  assert.deepEqual(
    [...support.availableModelHubBenchmarkProfiles(counts, 'bench@1')].sort(),
    ['agent', 'independent'],
  )
  assert.deepEqual(
    [...support.availableModelHubBenchmarkMetrics(counts, 'bench@1', 'agent')],
    ['score'],
  )
  assert.deepEqual(
    support.preferredModelHubBenchmarkSelection(
      counts,
      'bench@1',
      'independent',
    ),
    { benchmark: 'bench@1', profile: 'independent', metric: 'lower_bound' },
  )
})

test('website benchmark colors are stable and collision-free for visible models', () => {
  const modelIDs = [
    'meta/muse-glimmer-30b',
    'meta/llama-4-maverick-17b-128e-instruct',
    'ai21/jamba-reasoning-3b',
    'mistral/mistral-medium-3.5',
  ]
  const hues = support.modelHubChartHues(modelIDs)

  assert.equal(new Set(hues.values()).size, modelIDs.length)
  assert.deepEqual(
    [...support.modelHubChartHues([...modelIDs].reverse())],
    [...hues],
  )
})

test('website evaluation details label every metric value', () => {
  const page = readFileSync(
    resolve(repositoryRoot, 'website/src/pages/models.tsx'),
    'utf8',
  )

  assert.match(page, /<small>\{readable\(id\)\}<\/small>/)
  assert.match(page, /<b>[\s\S]*?formatMetric\(value, definition\)/)
})

test('website model table uses a native keyboard target without scrolling on Space', () => {
  const page = readFileSync(
    resolve(repositoryRoot, 'website/src/pages/models.tsx'),
    'utf8',
  )
  const table = page.slice(
    page.indexOf('function ModelTable'),
    page.indexOf('function BenchmarkBar'),
  )

  assert.match(table, /<button[\s\S]*?styles\.tableModelButton/)
  assert.match(table, /if \(event\.key === ' '\) event\.preventDefault\(\)/)
  assert.doesNotMatch(table, /<tr[\s\S]{0,120}?tabIndex=/)
})

test('website virtual model detail resolves the complete backend pool', () => {
  const page = readFileSync(
    resolve(repositoryRoot, 'website/src/pages/models.tsx'),
    'utf8',
  )

  assert.match(page, /<small>Entrypoint<\/small>/)
  assert.match(page, /model\.entrypoint \?\? model\.id/)
  assert.match(page, /const candidateModel = modelByID\.get\(candidate\)/)
  assert.match(page, /presentation=\{candidateModel\.presentation\}/)
  assert.match(page, /'Custom model slot'/)
})
