import { describe, expect, it } from 'vitest'

import generatedCatalog from '../modelCatalogDocument'
import type {
  BuiltInModelCatalog,
  CatalogBenchmark,
  CatalogEvaluation,
} from '../types/modelCatalog'
import { buildEffectiveEvaluationGroups } from './configPageEffectiveEvaluations'
import type { ConfigData, EvaluationRecordConfig, NormalizedModel } from './configPageSupport'

const benchmark: CatalogBenchmark = {
  id: 'acme/quality@1.0.0',
  display_name: 'Quality',
  domain: 'general',
  default_profile: 'standard',
  profiles: [{ id: 'standard', display_name: 'Standard', description: '' }],
  metrics: [{ id: 'accuracy', unit: 'ratio', direction: 'higher_is_better', range: [0, 1] }],
}
const evidence: CatalogEvaluation = {
  id: 'acme/published',
  model: 'acme/model',
  benchmark: benchmark.id,
  benchmark_profile: 'standard',
  reasoning_effort: 'high',
  subject: { dataset: 'original' },
  metrics: { accuracy: 0.8 },
  status: 'available',
  measured_at: '2026-09-01',
  evidence: {
    provenance: 'vendor_claimed',
    verification: 'imported',
    source: 'https://example.com/model',
    redistributable: true,
  },
}
const model: NormalizedModel = { name: 'production', catalog: 'acme/model', endpoints: [] }
const catalog: BuiltInModelCatalog = {
  schema_version: 'vllm-sr/model-catalog/v2',
  catalogs: [],
  protocols: [],
  providers: [],
  reasoning_families: [],
  models: [],
  benchmarks: [benchmark],
  evaluations: [evidence],
  indices: [],
  index_results: [],
}
const configured: EvaluationRecordConfig = {
  model: 'acme/model',
  benchmark: benchmark.id,
  metrics: { accuracy: 0.75 },
  source: 'https://example.com/measurement',
  measured_at: '2026-09-02',
  metadata: { batch: 8, verified: false },
}
const configWith = (records: EvaluationRecordConfig[], benchmarks?: CatalogBenchmark[]) =>
  ({ evaluation: { records, benchmarks } }) as ConfigData

describe('effective model evaluation evidence', () => {
  it('combines built-in and configured evidence under a configured alias without mutating inputs', () => {
    const config = configWith([configured])
    const before = JSON.stringify({ config, catalog })
    const [group] = buildEffectiveEvaluationGroups(config, [model], catalog)
    expect(group).toMatchObject({
      modelName: 'production',
      catalogId: 'acme/model',
      benchmarkCount: 1,
      availableCount: 2,
      builtInCount: 1,
      configuredCount: 1,
    })
    expect(group.records[0].evaluation).toEqual(evidence)
    expect(group.records[1].evaluation).toMatchObject({
      benchmark_profile: 'standard',
      reasoning_effort: 'default',
      subject: { parameters: configured.metadata },
      evidence: { provenance: 'operator', verification: 'claimed', source: configured.source },
    })
    expect(group.records[1].configuredRecord).toEqual(configured)
    expect(JSON.stringify({ config, catalog })).toBe(before)
  })

  it('does not inherit catalog evidence for an unbound custom model with the same name', () => {
    const [group] = buildEffectiveEvaluationGroups(
      configWith([configured]),
      [{ name: 'acme/model', endpoints: [] }],
      catalog,
    )
    expect(group.builtInCount).toBe(0)
    expect(group.configuredCount).toBe(1)
  })

  it('preserves conflicting available values and flags both instead of replacing or averaging', () => {
    const [group] = buildEffectiveEvaluationGroups(
      configWith([{ ...configured, reasoning_effort: 'high' }]),
      [model],
      catalog,
    )
    expect(group.records.map((record) => record.evaluation.metrics.accuracy)).toEqual([0.8, 0.75])
    expect(
      group.records.every((record) =>
        record.issues.some((issue) => issue.includes('Duplicate available metric')),
      ),
    ).toBe(true)
    expect(new Set(group.records.map((record) => record.key)).size).toBe(2)
    expect(evidence).not.toHaveProperty('issues')
  })

  it('keeps profiles, efforts and unavailable evidence separate without inventing zero scores', () => {
    const variants: CatalogEvaluation[] = [
      evidence,
      { ...evidence, id: 'another-profile', benchmark_profile: 'extended' },
      { ...evidence, id: 'missing', status: 'missing', metrics: {} },
    ]
    const [group] = buildEffectiveEvaluationGroups(configWith([configured]), [model], {
      ...catalog,
      evaluations: variants,
    })
    expect(group.records).toHaveLength(4)
    expect(group.availableCount).toBe(3)
    expect(group.records.every((record) => record.issues.length === 0)).toBe(true)
    expect(group.records[2].evaluation.metrics).toEqual({})
  })

  it('retains unknown benchmarks as unindexed and resolves newly declared benchmark defaults', () => {
    const customBenchmark = {
      ...benchmark,
      id: 'team/custom@1.0.0',
      default_profile: 'custom-profile',
    }
    const unknown = { ...configured, benchmark: 'team/unknown@1.0.0' }
    const known = { ...configured, benchmark: customBenchmark.id }
    const [group] = buildEffectiveEvaluationGroups(
      configWith([unknown, known], [customBenchmark]),
      [model],
      catalog,
    )
    expect(group.records[1].issues).toEqual([expect.stringContaining('not used in scoring')])
    expect(group.records[1].benchmark).toBeUndefined()
    expect(group.records[2].evaluation.benchmark_profile).toBe('custom-profile')
    expect(group.records[2].benchmark).toEqual(customBenchmark)
    expect(group.records[2].issues).toEqual([])
  })

  it('keeps models without evidence visible and handles unloaded config/catalog', () => {
    const [group] = buildEffectiveEvaluationGroups(null, [model], null)
    expect(group).toMatchObject({
      modelName: 'production',
      records: [],
      availableCount: 0,
      builtInCount: 0,
      configuredCount: 0,
      benchmarkCount: 0,
    })
    expect(buildEffectiveEvaluationGroups(null, [], null)).toEqual([])
  })

  it('flags duplicate benchmark definitions without replacing built-in semantics', () => {
    const conflicting = { ...benchmark, default_profile: 'changed' }
    const [group] = buildEffectiveEvaluationGroups(
      configWith([configured], [conflicting]),
      [model],
      catalog,
    )
    expect(
      group.records.every((record) =>
        record.issues.some((issue) => issue.includes('Duplicate benchmark definition')),
      ),
    ).toBe(true)
    expect(group.records[1].evaluation.benchmark_profile).toBe('standard')
    expect(group.records[1].evaluation.id).toBe('operator/0/acme-model')
    expect(group.records[1].benchmark).toEqual(benchmark)
  })

  it('scopes real packaged evidence to configured model identities, including aliases', () => {
    const bundled = generatedCatalog as unknown as BuiltInModelCatalog
    const ids = ['zai/glm-5.3-flash', 'qwen/qwen3.8-flash-next', 'qwen/qwen3.8-27b']
    const groups = buildEffectiveEvaluationGroups(
      null,
      ids.map((id, index) => ({ name: `serving-${index}`, catalog: id, endpoints: [] })),
      bundled,
    )
    expect(groups).toHaveLength(3)
    groups.forEach((group, index) => {
      const expected = bundled.evaluations.filter((item) => item.model === ids[index])
      expect(expected.length).toBeGreaterThan(0)
      expect(group.records.map((record) => record.evaluation)).toEqual(expected)
      expect(group.configuredCount).toBe(0)
    })
  })
})
