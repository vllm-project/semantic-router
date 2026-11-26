import { describe, expect, it } from 'vitest'

import generatedCatalog from '../modelCatalogDocument'
import type { BuiltInModelCatalog } from '../types/modelCatalog'
import {
  modelHubBenchmarkNormalizedValue,
  modelHubBenchmarkValueLabel,
} from './modelHubBenchmarkNormalization'
import {
  MODEL_HUB_DEFAULT_FILTERS,
  formatContextWindow,
  modelHubCapabilities,
  modelHubContextLabel,
  modelHubCreators,
  modelHubEvaluationConditionLabel,
  modelHubMaxOutputLabel,
  modelHubParameterLabel,
  modelHubProviders,
  modelHubPublicEvaluations,
  modelHubRows,
  modelHubStats,
  type ModelHubFilters,
} from './modelHubSupport'

const catalog = generatedCatalog as unknown as BuiltInModelCatalog
const filters = (patch: Partial<ModelHubFilters> = {}): ModelHubFilters => ({
  ...MODEL_HUB_DEFAULT_FILTERS,
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
    const physicalCreators = new Set(
      catalog.models.filter((model) => model.kind === 'physical').map((model) => model.publisher),
    )
    expect(stats.models).toBe(catalog.models.length)
    expect(stats.providers).toBe(catalog.providers.length)
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
      filters({ query: 'glm-5.3', distribution: 'open_weights', publishers: ['Z.ai / GLM'] }),
    )
    expect(rows.map((row) => row.model.id)).toEqual(['zai/glm-5.3', 'zai/glm-5.3-flash'])
    expect(rows.every((row) => row.providers.length > 0)).toBe(true)
    const capability = modelHubCapabilities(catalog)[0]
    expect(
      modelHubRows(catalog, filters({ capabilities: [capability] })).every((row) =>
        row.model.capabilities.includes(capability),
      ),
    ).toBe(true)
    const provider = modelHubProviders(catalog)[0]
    expect(
      modelHubRows(catalog, filters({ providers: [provider.id] })).every((row) =>
        row.providers.some((entry) => entry.provider.id === provider.id),
      ),
    ).toBe(true)
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
  it('orders Gemini 4 by official GA dates ahead of earlier Gemini releases', () => {
    const rows = modelHubRows(catalog, filters({ query: 'gemini', sort: 'released' }))
    expect(rows.slice(0, 4).map((row) => [row.model.id, row.model.released_at])).toEqual([
      ['google/gemini-4-neon', '2026-11-18'],
      ['google/gemini-4-hydrogen', '2026-11-12'],
      ['google/gemini-4-carbon', '2026-11-10'],
      ['google/gemini-4-argon', '2026-10-13'],
    ])
  })

  it('sorts catalog properties without turning the hub into a composite ranking', () => {
    const byContext = modelHubRows(catalog, filters({ sort: 'context' }))
    expect(byContext[0].model.limits?.context_window_size ?? 0).toBeGreaterThanOrEqual(
      byContext[1].model.limits?.context_window_size ?? 0,
    )
    const byProvider = modelHubRows(catalog, filters({ sort: 'providers' }))
    expect(byProvider[0].providers.length).toBeGreaterThanOrEqual(byProvider[1].providers.length)
    const byRelease = modelHubRows(catalog, filters({ sort: 'released' }))
    expect((byRelease[0].model.released_at ?? '') >= (byRelease[1].model.released_at ?? '')).toBe(
      true,
    )
  })

  it('treats active and experimental cards as the supported catalog surface', () => {
    const supported = modelHubRows(catalog, filters())
    expect(supported.length).toBeGreaterThan(0)
    expect(
      supported.every(
        (row) => row.model.lifecycle === 'active' || row.model.lifecycle === 'experimental',
      ),
    ).toBe(true)
    expect(modelHubRows(catalog, filters({ lifecycle: 'all' })).length).toBeGreaterThanOrEqual(
      supported.length,
    )
  })

  it('formats context windows explicitly', () => {
    expect(formatContextWindow(1_000_000)).toBe('1M')
    expect(formatContextWindow(1_048_576)).toBe('1M')
    expect(formatContextWindow(1_050_000)).toBe('1.05M')
    expect(formatContextWindow(131_072)).toBe('128K')
    expect(formatContextWindow(128_000)).toBe('128K')
    expect(formatContextWindow()).toBe('Not published')
  })

  it('distinguishes disclosed values from provider and runtime-owned metadata', () => {
    const gemini = catalog.models.find((model) => model.id === 'google/gemini-3.8-flash')!
    const openModel = catalog.models.find((model) => model.id === 'openai/gpt-oss-20b')!

    expect(modelHubContextLabel(gemini)).toBe('1M')
    expect(modelHubMaxOutputLabel(gemini)).toBe('64K')
    expect(modelHubParameterLabel(gemini)).toBe('Undisclosed')
    expect(modelHubMaxOutputLabel(openModel)).toBe('Runtime-set')
  })
})

describe('model hub benchmark evidence', () => {
  it('includes Gemini 4 in the existing public AA GDP.pdf comparison', () => {
    const records = modelHubPublicEvaluations(catalog).filter(
      (evaluation) =>
        evaluation.benchmark === 'artificial-analysis/gdp-pdf@1.0.0' &&
        evaluation.benchmark_profile === 'published-standard' &&
        evaluation.metrics.all_pass !== undefined,
    )
    expect(records.map((evaluation) => evaluation.model)).toEqual(
      expect.arrayContaining([
        'google/gemini-3.8-flash',
        'google/gemini-4-argon',
        'google/gemini-4-carbon',
        'google/gemini-4-hydrogen',
        'google/gemini-4-neon',
      ]),
    )
  })

  it('does not use August GameBench models to qualify November Astra and Sol scores', () => {
    const gameBench = catalog.evaluations.filter(
      (row) => row.benchmark === 'duellab/gamebench@2.0.0',
    )
    const august = gameBench.filter(
      (row) => row.subject.benchmark_release === 'gb2-20260824-e05a1ff89a26',
    )
    const models = new Set(['openai/gpt-6.2-astra', 'openai/gpt-6.2-sol'])
    const november = gameBench.filter(
      (row) =>
        models.has(row.model) && row.subject.benchmark_release === 'gb2-20261119-235af8c04037',
    )
    expect(new Set(november.map((row) => row.model))).toEqual(models)
    expect(new Set(august.map((row) => row.model)).size).toBeGreaterThanOrEqual(10)

    const visible = modelHubPublicEvaluations({ ...catalog, evaluations: [...august, ...november] })
    expect(visible.map((row) => row.id)).toEqual(august.map((row) => row.id))
  })

  it.each([
    ['gpt-6.2-astra', 'xhigh', 0.28],
    ['gpt-6.2-sol', 'xhigh', 0.284],
    ['gpt-6.1-terra', 'high', 0.28],
  ] as const)('finds %s on OpenAI with its measured GDP.pdf effort', (id, effort, score) => {
    const rows = modelHubRows(catalog, filters({ query: id, providers: ['openai'] }))
    expect(rows.map((row) => row.model.id)).toEqual([`openai/${id}`])
    const evidence = modelHubPublicEvaluations(catalog).filter(
      (record) =>
        record.model === rows[0].model.id && record.benchmark === 'artificial-analysis/gdp-pdf@1.0.0',
    )
    expect(evidence.map((record) => [record.reasoning_effort, record.metrics.all_pass])).toEqual([
      [effort, score],
    ])
  })

  it('presents Elo as a normalized percentage without replacing the raw measurement', () => {
    const metric = catalog.benchmarks
      .find((benchmark) => benchmark.id === 'artificial-analysis/gdpval-aa@2.0.0')!
      .metrics.find((candidate) => candidate.id === 'elo')!

    expect(modelHubBenchmarkNormalizedValue(1769.1, metric)).toBeCloseTo(0.63455)
    expect(modelHubBenchmarkValueLabel(1769.1, metric)).toBe('63.5%')
    expect(metric.unit).toBe('elo')
  })

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
})
