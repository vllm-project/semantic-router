import { describe, expect, it } from 'vitest'

import generatedCatalog from '../modelCatalogDocument'
import type { BuiltInModelCatalog } from '../types/modelCatalog'
import {
  modelHubBenchmarkNormalizedValue,
  modelHubBenchmarkValueLabel,
} from './modelHubBenchmarkNormalization'
import {
  benchmarkName,
  formatContextWindow,
  formatIntelligence,
  modelHubBenchmarkBarHeight,
  modelHubBenchmarkDomain,
  modelHubBenchmarkOverviewSelections,
  modelHubBenchmarkPoints,
  modelHubBenchmarkSelections,
  modelHubChartColors,
  modelHubChartColorToken,
  modelHubCreators,
  modelHubDefaultBenchmark,
  modelHubEvaluationConditionLabel,
  modelHubProviders,
  modelHubContextLabel,
  modelHubMaxOutputLabel,
  modelHubParameterLabel,
  modelHubPageForModel,
  modelHubPublicEvaluations,
  modelHubRows,
  modelHubStats,
  paginateModelHubRows,
  resolveModelHubSelection,
  type ModelHubFilters,
} from './modelHubSupport'

const catalog = generatedCatalog as unknown as BuiltInModelCatalog
const filters = (patch: Partial<ModelHubFilters> = {}): ModelHubFilters => ({
  query: '',
  kind: 'all',
  distribution: 'all',
  lifecycle: 'supported',
  publisher: 'all',
  provider: 'all',
  capability: 'all',
  sort: 'name',
  ...patch,
})

describe('model hub inventory support', () => {
  it('distinguishes configurable reasoning effort from published run conditions', () => {
    const configurable = { reasoning_family: 'openai-reasoning' }
    const published = {}

    expect(modelHubEvaluationConditionLabel(configurable, 'high')).toBe('high effort')
    expect(modelHubEvaluationConditionLabel(published, 'unspecified')).toBe('Effort not reported')
    expect(modelHubEvaluationConditionLabel(published, 'default')).toBe('Published default')
    expect(modelHubEvaluationConditionLabel(published, 'enabled')).toBe('Reasoning enabled')
    expect(modelHubEvaluationConditionLabel(published, 'disabled')).toBe('Non-reasoning run')
    expect(modelHubEvaluationConditionLabel(published, 'adaptive')).toBe('Adaptive reasoning')
    expect(modelHubEvaluationConditionLabel(published, 'none')).toBe('No reasoning')
    expect(modelHubEvaluationConditionLabel(published, 'no_think')).toBe('No reasoning')
    expect(modelHubEvaluationConditionLabel(published, 'custom_profile')).toBe('Custom profile run')
  })

  it('projects the complete generated inventory without maintaining another list', () => {
    const stats = modelHubStats(catalog)
    const physicalModels = catalog.models.filter((model) => model.kind === 'physical')
    const virtualModels = catalog.models.filter((model) => model.kind === 'virtual')
    const physicalCreators = new Set(physicalModels.map((model) => model.publisher))
    expect(stats.models).toBe(catalog.models.length)
    expect(stats.physicalModels).toBe(physicalModels.length)
    expect(stats.virtualModels).toBe(virtualModels.length)
    expect(stats.mappedProviders).toBe(modelHubProviders(catalog).length)
    expect(stats.providerContracts).toBe(catalog.providers.length)
    expect(stats.creators).toBe(physicalCreators.size)
    expect(physicalCreators.size).toBeGreaterThanOrEqual(20)
    expect(modelHubCreators(catalog)).toEqual(
      [...physicalCreators].sort((left, right) => left.localeCompare(right)),
    )
    expect(modelHubCreators(catalog)).not.toContain('vllm-sr.ai')
  })

  it('searches and filters canonical model metadata', () => {
    const rows = modelHubRows(
      catalog,
      filters({
        query: 'glm-5.3',
        kind: 'physical',
        distribution: 'open_weights',
        publisher: 'Z.ai / GLM',
      }),
    )
    expect(rows.map((row) => row.model.id)).toEqual(['zai/glm-5.3', 'zai/glm-5.3-flash'])
    expect(rows.every((row) => row.providers.length > 0)).toBe(true)
  })

  it('offers only serving providers that bind at least one Hub model', () => {
    const modelIDs = new Set(catalog.models.map((model) => model.id))
    const providers = modelHubProviders(catalog)

    expect(providers.length).toBeGreaterThan(0)
    expect(providers.length).toBeLessThan(catalog.providers.length)
    expect(
      providers.every((provider) =>
        provider.models?.some((binding) => modelIDs.has(binding.catalog)),
      ),
    ).toBe(true)
  })
})

describe('model hub presentation support', () => {
  it('sorts catalog properties without turning the hub into a composite ranking', () => {
    const byContext = modelHubRows(catalog, filters({ kind: 'physical', sort: 'context' }))
    expect(byContext[0].model.limits?.context_window_size ?? 0).toBeGreaterThanOrEqual(
      byContext[1].model.limits?.context_window_size ?? 0,
    )
    const byProvider = modelHubRows(catalog, filters({ kind: 'physical', sort: 'providers' }))
    expect(byProvider[0].providers.length).toBeGreaterThanOrEqual(byProvider[1].providers.length)
  })

  it('treats active and experimental cards as the supported catalog surface', () => {
    const supported = modelHubRows(catalog, filters({ kind: 'physical' }))
    expect(supported.length).toBeGreaterThan(0)
    expect(
      supported.every(
        (row) => row.model.lifecycle === 'active' || row.model.lifecycle === 'experimental',
      ),
    ).toBe(true)
  })

  it('formats context, missing scores, and benchmark labels explicitly', () => {
    expect(formatContextWindow(1_000_000)).toBe('1M')
    expect(formatContextWindow(1_048_576)).toBe('1M')
    expect(formatContextWindow(1_050_000)).toBe('1.05M')
    expect(formatContextWindow(131_072)).toBe('128K')
    expect(formatContextWindow(128_000)).toBe('128K')
    expect(formatContextWindow()).toBe('Not published')
    expect(formatIntelligence(null)).toBe('Not yet measured')
    expect(benchmarkName('idavidrein/gpqa-diamond@1.0.0', 'accuracy', catalog)).toBe(
      'GPQA Diamond · accuracy',
    )
  })

  it('distinguishes disclosed values from provider and runtime-owned metadata', () => {
    const gemini = catalog.models.find((model) => model.id === 'google/gemini-3.8-flash')!
    const openModel = catalog.models.find((model) => model.id === 'openai/gpt-oss-20b')!

    expect(modelHubContextLabel(gemini)).toBe('1M')
    expect(modelHubMaxOutputLabel(gemini)).toBe('64K')
    expect(modelHubParameterLabel(gemini)).toBe('Undisclosed')
    expect(modelHubMaxOutputLabel(openModel)).toBe('Runtime-set')
  })

  it('paginates large inventories with stable, bounded page metadata', () => {
    const rows = modelHubRows(catalog, filters())
    const first = paginateModelHubRows(rows, 1, 24)
    const beyondEnd = paginateModelHubRows(rows, 100_000, 24)
    expect(first.items).toHaveLength(24)
    expect(first.start).toBe(1)
    expect(first.end).toBe(24)
    expect(beyondEnd.page).toBe(beyondEnd.totalPages)
    expect(beyondEnd.end).toBe(rows.length)
  })

  it('assigns stable, collision-free chart colors to the visible models', () => {
    const modelIDs = [
      'ai21/jamba-reasoning-3b',
      'google/gemma-4-31b-it',
      'thinking-machines/inkling',
      'anthropic/claude-fable-5.1',
      'anthropic/claude-opus-5',
    ]
    const colors = modelHubChartColors(modelIDs)
    const tokens = [...colors.values()].map(modelHubChartColorToken)

    expect(new Set(tokens)).toHaveLength(modelIDs.length)
    expect(modelHubChartColors([...modelIDs].reverse())).toEqual(colors)
  })

  it('keeps a model color stable when its chart peers change', () => {
    const qwen = 'qwen/qwen3.7-max'
    const universe = catalog.models.map((model) => model.id)
    const alone = modelHubChartColors([qwen], universe).get(qwen)
    const withPeer = modelHubChartColors(['anthropic/claude-opus-4.8', qwen], universe).get(qwen)

    expect(withPeer).toStrictEqual(alone)
  })

  it('assigns unique stable colors to every current catalog model', () => {
    const modelIDs = catalog.models.map((model) => model.id)
    const colors = modelHubChartColors(modelIDs)
    const tokens = new Map([...colors].map(([id, color]) => [id, modelHubChartColorToken(color)]))

    expect(new Set(tokens.values())).toHaveLength(modelIDs.length)
    expect(tokens.get('anthropic/claude-opus-4.8')).not.toBe(tokens.get('qwen/qwen3.7-max'))
    expect(tokens.get('qwen/qwen3.8-max')).not.toBe(tokens.get('google/gemini-3.8-flash'))
    expect(tokens.get('nvidia/nemotron-3-ultra')).not.toBe(tokens.get('stepfun/step-3.7-flash'))
    expect(tokens.get('tencent/hy3')).not.toBe(tokens.get('bytedance/seed-2.0-pro'))
    expect(tokens.get('moonshot/kimi-k2.5')).not.toBe(tokens.get('moonshot/kimi-k2.6'))
  })
})

describe('model hub benchmark presentation contract', () => {
  it('presents Elo as a normalized percentage without replacing the raw measurement', () => {
    const metric = catalog.benchmarks
      .find((benchmark) => benchmark.id === 'artificial-analysis/gdpval-aa@2.0.0')!
      .metrics.find((candidate) => candidate.id === 'elo')!

    expect(modelHubBenchmarkNormalizedValue(1769.1, metric)).toBeCloseTo(0.63455)
    expect(modelHubBenchmarkValueLabel(1769.1, metric)).toBe('63.5%')
    expect(metric.unit).toBe('elo')
  })

  it('defines the six core comparisons in catalog data', () => {
    expect(
      catalog.benchmarks
        .filter((benchmark) => benchmark.tags?.includes('core'))
        .map((benchmark) => benchmark.display_name),
    ).toEqual([
      'MMLU-Pro',
      'GPQA Diamond',
      "Humanity's Last Exam",
      'SWE-bench Verified',
      'Terminal-Bench 2.1',
      'SciCode',
    ])
  })
})

describe('model hub selection and benchmark support', () => {
  it('omits narrow benchmark evidence from every public Hub surface', () => {
    const broad = Array.from({ length: 10 }, (_, index) => ({
      id: `broad-${index}`,
      model: `model-${index}`,
      benchmark: 'broad@1',
      benchmark_profile: 'default',
      reasoning_effort: 'default',
      status: 'available' as const,
      metrics: { score: 0.8 },
      subject: {},
      evidence: {
        provenance: 'vendor_claimed' as const,
        verification: 'claimed' as const,
        redistributable: false,
      },
    }))
    const narrow = broad.slice(0, 9).map((evaluation, index) => ({
      ...evaluation,
      id: `narrow-${index}`,
      benchmark: 'narrow@1',
    }))
    const filtered = modelHubPublicEvaluations({
      ...catalog,
      evaluations: [...broad, ...narrow],
    })

    expect(filtered).toHaveLength(10)
    expect(new Set(filtered.map((evaluation) => evaluation.benchmark))).toEqual(
      new Set(['broad@1']),
    )
  })

  it('scales the best observed result to the full chart height', () => {
    const metric = {
      id: 'score',
      unit: 'proportion',
      direction: 'higher_is_better' as const,
      range: [0, 1] as [number, number],
    }
    const domain = modelHubBenchmarkDomain([0.91, 0.65, 0.26], metric)

    expect(domain).toEqual([0, 0.91])
    expect(modelHubBenchmarkBarHeight(0.91, ...domain, metric.direction)).toBe(100)
    expect(modelHubBenchmarkBarHeight(0.65, ...domain, metric.direction)).toBeCloseTo(
      (0.65 / 0.91) * 100,
    )
  })

  it('keeps table and card selection on the visible page after pagination or filtering', () => {
    const rows = modelHubRows(catalog, filters())
    const firstPage = paginateModelHubRows(rows, 1, 24)
    const secondPage = paginateModelHubRows(rows, 2, 24)
    const selectedOnFirstPage = firstPage.items[3]

    expect(resolveModelHubSelection(firstPage.items, selectedOnFirstPage.model.id)).toBe(
      selectedOnFirstPage,
    )
    expect(resolveModelHubSelection(secondPage.items, selectedOnFirstPage.model.id)).toBe(
      secondPage.items[0],
    )

    const filtered = modelHubRows(catalog, filters({ publisher: 'Z.ai / GLM' }))
    expect(resolveModelHubSelection(filtered, selectedOnFirstPage.model.id)).toBe(filtered[0])
  })

  it('builds benchmark-specific model-effort points from available records only', () => {
    const selection = modelHubDefaultBenchmark(catalog)
    expect(selection).not.toBeNull()
    const selections = modelHubBenchmarkSelections(catalog)
    expect(selections.every((candidate) => candidate.modelCount >= 10)).toBe(true)
    const expectedDefault = selections[0]
    expect(`${selection!.benchmark}\u0000${selection!.profile}\u0000${selection!.metric}`).toBe(
      `${expectedDefault.benchmark}\u0000${expectedDefault.profile}\u0000${expectedDefault.metric}`,
    )
    const points = modelHubBenchmarkPoints(catalog, selection!)
    expect(points.length).toBeGreaterThan(0)
    expect(points.every((point) => Number.isFinite(point.value))).toBe(true)
    expect(new Set(points.map((point) => point.reasoningEffort)).size).toBeGreaterThan(0)
    const metric = catalog.benchmarks
      .find((benchmark) => benchmark.id === selection!.benchmark)
      ?.metrics.find((candidate) => candidate.id === selection!.metric)
    const expected = [...points.map((point) => point.value)].sort((left, right) =>
      metric?.direction === 'lower_is_better' ? left - right : right - left,
    )
    expect(points.map((point) => point.value)).toEqual(expected)
    expect(
      points.every((point) =>
        catalog.evaluations.some(
          (evaluation) =>
            evaluation.id === point.evaluation &&
            evaluation.model === point.model.id &&
            evaluation.reasoning_effort === point.reasoningEffort &&
            evaluation.status === 'available',
        ),
      ),
    ).toBe(true)
  })
})

describe('model hub cross-view navigation', () => {
  it('routes a benchmark bar back to the model page that owns its card', () => {
    const rows = modelHubRows(catalog, filters())
    const target = rows[27]

    expect(modelHubPageForModel(rows, target.model.id, 10)).toBe(3)
    expect(modelHubPageForModel(rows, 'missing/model', 10)).toBe(1)
  })

  it('chooses one broad comparison card per benchmark for the stacked explorer', () => {
    const overview = modelHubBenchmarkOverviewSelections(catalog)

    expect(new Set(overview.map((selection) => selection.benchmark))).toHaveLength(overview.length)
    expect(overview.every((selection) => selection.modelCount >= 10)).toBe(true)
  })
})
