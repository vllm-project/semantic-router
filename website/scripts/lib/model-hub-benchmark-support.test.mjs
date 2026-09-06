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

test('website benchmark controls expose only available exact tuples', () => {
  const counts = support.modelHubBenchmarkSelectionCounts([
    {
      status: 'available',
      model: 'acme/agent',
      benchmark: 'bench@1',
      benchmark_profile: 'agent',
      metrics: { score: 0.8, lower_bound: null },
    },
    {
      status: 'available',
      model: 'acme/independent',
      benchmark: 'bench@1',
      benchmark_profile: 'independent',
      metrics: { score: 0.9, lower_bound: 0.85 },
    },
    {
      status: 'missing',
      model: 'acme/missing',
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

test('website benchmark colors are stable and collision-free for visible models', () => {
  const modelIDs = [
    'meta/muse-glimmer-30b',
    'meta/llama-4-maverick-17b-128e-instruct',
    'ai21/jamba-reasoning-3b',
    'mistral/mistral-medium-3.5',
  ]
  const universe = catalog.models.map(model => model.id)
  const colors = support.modelHubChartColors(modelIDs, universe)
  const tokens = [...colors.values()].map(support.modelHubChartColorToken)

  assert.equal(new Set(tokens).size, modelIDs.length)
  assert.deepEqual(
    [...support.modelHubChartColors([...modelIDs].reverse(), universe)],
    [...colors],
  )
})

test('website benchmark colors do not depend on the visible peer set', () => {
  const qwen = 'qwen/qwen3.7-max'
  const universe = catalog.models.map(model => model.id)
  const alone = support.modelHubChartColors([qwen], universe).get(qwen)
  const withPeer = support
    .modelHubChartColors(['anthropic/claude-opus-4.8', qwen], universe)
    .get(qwen)

  assert.deepEqual(withPeer, alone)
  assert.notEqual(
    support.modelHubChartColorToken(
      support.modelHubChartColors(['anthropic/claude-opus-4.8'], universe).get(
        'anthropic/claude-opus-4.8',
      ),
    ),
    support.modelHubChartColorToken(alone),
  )
})

test('website benchmark palette distinguishes the complete catalog and real chart peers', () => {
  const modelIDs = catalog.models.map(model => model.id)
  const colors = support.modelHubChartColors(modelIDs)
  const tokens = new Map(
    [...colors].map(([id, color]) => [id, support.modelHubChartColorToken(color)]),
  )

  assert.equal(new Set(tokens.values()).size, modelIDs.length)
  for (const [left, right] of [
    ['qwen/qwen3.8-max', 'google/gemini-3.8-flash'],
    ['nvidia/nemotron-3-ultra', 'stepfun/step-3.7-flash'],
    ['tencent/hy3', 'bytedance/seed-2.0-pro'],
    ['moonshot/kimi-k2.5', 'moonshot/kimi-k2.6'],
  ]) {
    assert.notEqual(tokens.get(left), tokens.get(right), `${left} / ${right}`)
  }
})

test('every published benchmark comparison keeps different models perceptually distinct', () => {
  const modelIDs = catalog.models.map(model => model.id)
  const colors = support.modelHubChartColors(modelIDs)
  const metrics = new Map(
    catalog.benchmarks.flatMap(benchmark =>
      benchmark.metrics.map(metric => [`${benchmark.id}\u0000${metric.id}`, metric]),
    ),
  )
  const tuples = new Map()
  for (const evaluation of catalog.evaluations) {
    if (evaluation.status !== 'available') continue
    for (const [metric, value] of Object.entries(evaluation.metrics ?? {})) {
      if (typeof value !== 'number') continue
      const key = `${evaluation.benchmark}\u0000${evaluation.benchmark_profile}\u0000${metric}`
      tuples.set(key, [...(tuples.get(key) ?? []), { model: evaluation.model, value }])
    }
  }

  for (const [key, rows] of tuples) {
    const [benchmark, , metricID] = key.split('\u0000')
    const metric = metrics.get(`${benchmark}\u0000${metricID}`)
    rows.sort((left, right) =>
      metric?.direction === 'lower_is_better'
        ? left.value - right.value
        : right.value - left.value,
    )
    for (let left = 0; left < rows.length; left += 1) {
      for (let right = left + 1; right < rows.length; right += 1) {
        if (rows[left].model === rows[right].model) continue
        const distance = support.modelHubChartColorDistance(
          colors.get(rows[left].model),
          colors.get(rows[right].model),
        )
        assert.ok(
          distance >= 0.045,
          `${key}: ${rows[left].model} / ${rows[right].model} (${distance})`,
        )
      }
    }
  }
})

test('website benchmark renders every filtered result in one comparison surface', () => {
  const component = readFileSync(
    resolve(repositoryRoot, 'website/src/components/model-hub/ModelHubBenchmark.tsx'),
    'utf8',
  )

  assert.match(component, /rows\.map\(row =>/)
  assert.match(component, /Benchmark comparison with all filtered results/)
  assert.doesNotMatch(component, /<Pagination/)
  assert.doesNotMatch(component, /pageRows/)
})

test('website benchmark bar height follows metric direction', () => {
  assert.ok(
    support.modelHubBenchmarkBarHeight(0.9, 0, 1, 'higher_is_better')
    > support.modelHubBenchmarkBarHeight(0.2, 0, 1, 'higher_is_better'),
  )
  assert.ok(
    support.modelHubBenchmarkBarHeight(100, 0, 100, 'lower_is_better')
    < support.modelHubBenchmarkBarHeight(10, 0, 100, 'lower_is_better'),
  )
})

test('website benchmark domain fills the chart from the best observed result', () => {
  const metric = {
    id: 'score',
    unit: 'proportion',
    direction: 'higher_is_better',
    range: [0, 1],
  }
  const domain = support.modelHubBenchmarkDomain([0.91, 0.65, 0.26], metric)

  assert.deepEqual(domain, [0, 0.91])
  assert.equal(support.modelHubBenchmarkBarHeight(0.91, ...domain, metric.direction), 100)
  assert.ok(
    Math.abs(
      support.modelHubBenchmarkBarHeight(0.65, ...domain, metric.direction)
      - (0.65 / 0.91) * 100,
    ) < 0.001,
  )
})

test('website evaluation details label every metric value', () => {
  const page = readFileSync(
    resolve(repositoryRoot, 'website/src/components/model-hub/ModelHubDetail.tsx'),
    'utf8',
  )

  assert.match(page, /<small>\{readable\(id\)\}<\/small>/)
  assert.match(page, /<b>[\s\S]*?formatMetric\(value, definition\)/)
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
