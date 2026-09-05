import { describe, expect, it } from 'vitest'

import generatedCatalog from '../generated/modelCatalog.json'
import type { BuiltInModelCatalog } from '../types/modelCatalog'
import {
  benchmarkName,
  formatContextWindow,
  formatIntelligence,
  modelHubBenchmarkPoints,
  modelHubChartHues,
  modelHubCreators,
  modelHubDefaultBenchmark,
  modelHubEvaluationConditionLabel,
  modelHubProviders,
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
    expect(formatContextWindow(131_072)).toBe('131.1K')
    expect(formatContextWindow()).toBe('Not published')
    expect(formatIntelligence(null)).toBe('Not yet measured')
    expect(benchmarkName('idavidrein/gpqa-diamond@1.0.0', 'accuracy', catalog)).toBe(
      'GPQA Diamond · accuracy',
    )
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

  it('assigns stable, collision-free chart hues to the visible models', () => {
    const modelIDs = [
      'ai21/jamba-reasoning-3b',
      'google/gemma-4-31b-it',
      'thinking-machines/inkling',
      'anthropic/claude-fable-5.1',
      'anthropic/claude-opus-5',
    ]
    const hues = modelHubChartHues(modelIDs)

    expect(new Set(hues.values())).toHaveLength(modelIDs.length)
    expect(modelHubChartHues([...modelIDs].reverse())).toEqual(hues)
  })
})

describe('model hub selection and benchmark support', () => {
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
    const exactSetupCounts = new Map<string, number>()
    for (const evaluation of catalog.evaluations) {
      if (evaluation.status !== 'available') continue
      for (const metric of Object.keys(evaluation.metrics)) {
        const key = `${evaluation.benchmark}\u0000${evaluation.benchmark_profile}\u0000${metric}`
        exactSetupCounts.set(key, (exactSetupCounts.get(key) ?? 0) + 1)
      }
    }
    const expectedDefault = Array.from(exactSetupCounts.entries()).sort(
      (left, right) => right[1] - left[1] || left[0].localeCompare(right[0]),
    )[0][0]
    expect(`${selection!.benchmark}\u0000${selection!.profile}\u0000${selection!.metric}`).toBe(
      expectedDefault,
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
