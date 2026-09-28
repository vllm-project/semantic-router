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
const catalog = JSON.parse(
  readFileSync(
    resolve(repositoryRoot, 'website/static/model-catalog/catalog.json'),
    'utf8',
  ),
)

test('website benchmark admission counts distinct models instead of effort rows', () => {
  const evaluations = Array.from({ length: 9 }, (_, index) => ({
    status: 'available',
    model: `acme/model-${index}`,
    benchmark: 'small@1',
    benchmark_profile: 'default',
    metrics: { score: 0.8 },
  }))
  evaluations.push({
    ...evaluations[0],
    metrics: { score: 0.9 },
  })

  assert.equal(
    support.modelHubBenchmarkSelectionCounts(evaluations, 10).size,
    0,
  )
  evaluations.push({
    ...evaluations[0],
    model: 'acme/model-9',
  })
  assert.equal(
    support.modelHubBenchmarkSelectionCounts(evaluations, 10).get(
      'small@1\u0000default\u0000score',
    ),
    10,
  )
})

test('website public evidence omits benchmarks with fewer than ten models', () => {
  const broad = Array.from({ length: 10 }, (_, index) => ({
    status: 'available',
    model: `acme/model-${index}`,
    benchmark: 'broad@1',
    benchmark_profile: 'default',
    metrics: { score: 0.8 },
  }))
  const narrow = broad.slice(0, 9).map(row => ({ ...row, benchmark: 'narrow@1' }))

  const visible = support.modelHubPublicEvaluations([...broad, ...narrow])
  assert.equal(visible.length, 10)
  assert.deepEqual(new Set(visible.map(row => row.benchmark)), new Set(['broad@1']))
})

test('website benchmark display normalizes Elo while retaining its raw unit', () => {
  const metric = catalog.benchmarks
    .find(benchmark => benchmark.id === 'artificial-analysis/gdpval-aa@2.0.0')
    .metrics.find(candidate => candidate.id === 'elo')

  assert.ok(Math.abs(support.modelHubBenchmarkNormalizedValue(1769.1, metric) - 0.63455) < 1e-9)
  assert.equal(support.modelHubBenchmarkRawValueLabel(1769.1, metric), '1769.10 elo')
  assert.equal(metric.unit, 'elo')
})

test('website catalog owns the exact six Intelligence 1.0 core benchmarks', () => {
  assert.deepEqual(
    catalog.benchmarks
      .filter(benchmark => benchmark.tags?.includes('core'))
      .map(benchmark => benchmark.display_name),
    [
      'MMLU-Pro',
      'GPQA Diamond',
      'Humanity\'s Last Exam',
      'Terminal-Bench 2.1',
      'SciCode',
      'LiveCodeBench v6',
    ],
  )
})

test('website evaluation details label every metric value', () => {
  const page = readFileSync(
    resolve(repositoryRoot, 'website/src/components/model-hub/ModelHubDetail.tsx'),
    'utf8',
  )

  assert.match(page, /<small>\{readable\(id\)\}<\/small>/)
  assert.match(page, /<b[\s\S]*?formatMetric\(value, definition\)/)
})

test('website model table uses a native keyboard target without scrolling on Space', () => {
  const page = readFileSync(
    resolve(repositoryRoot, 'website/src/components/model-hub/ModelHubDirectory.tsx'),
    'utf8',
  )
  const table = page.slice(
    page.indexOf('function ModelTable'),
    page.indexOf('function ViewToggle'),
  )

  assert.match(table, /<button[\s\S]*?styles\.tableModel/)
  assert.doesNotMatch(table, /<tr[\s\S]{0,120}?tabIndex=/)
})

test('website virtual model detail resolves the complete backend pool', () => {
  const page = readFileSync(
    resolve(repositoryRoot, 'website/src/components/model-hub/ModelHubDetail.tsx'),
    'utf8',
  )

  assert.match(page, /<small>Entrypoint<\/small>/)
  assert.match(page, /model\.entrypoint \?\? model\.id/)
  assert.match(page, /const candidateModel = modelByID\.get\(candidate\)/)
  assert.match(page, /presentation=\{candidateModel\.presentation\}/)
  assert.match(page, /'Custom model slot'/)
})
