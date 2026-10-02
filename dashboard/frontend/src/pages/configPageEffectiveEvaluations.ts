import type {
  BuiltInModelCatalog,
  CatalogBenchmark,
  CatalogEvaluation,
} from '../types/modelCatalog'
import type { ConfigData, EvaluationRecordConfig, NormalizedModel } from './configPageSupport'

export interface EffectiveEvaluationRecord {
  key: string
  origin: 'built_in' | 'configured'
  evaluation: CatalogEvaluation
  benchmark?: CatalogBenchmark
  issues: string[]
  configuredRecord?: EvaluationRecordConfig
}

export interface EffectiveEvaluationGroup {
  modelName: string
  catalogId: string
  model: NormalizedModel
  records: EffectiveEvaluationRecord[]
  benchmarkCount: number
  availableCount: number
  builtInCount: number
  configuredCount: number
}

function configuredEvidence(
  record: EvaluationRecordConfig,
  index: number,
  benchmark: CatalogBenchmark | undefined,
): EffectiveEvaluationRecord {
  return {
    key: `configured:${index}`,
    origin: 'configured',
    configuredRecord: record,
    benchmark,
    issues: benchmark
      ? []
      : ['Benchmark definition is missing; this record is retained but not used in scoring.'],
    evaluation: {
      id: `operator/${index}/${record.model.replace(/[/@:.]/g, '-').toLowerCase()}`,
      model: record.model,
      benchmark: record.benchmark,
      benchmark_profile: record.benchmark_profile || benchmark?.default_profile || '',
      reasoning_effort: record.reasoning_effort || 'default',
      subject: record.metadata ? { parameters: { ...record.metadata } } : {},
      metrics: { ...record.metrics },
      status: 'available',
      measured_at: record.measured_at,
      evidence: {
        provenance: 'operator',
        verification: 'claimed',
        source: record.source,
        redistributable: true,
      },
    },
  }
}

function flagConflictingMetrics(records: EffectiveEvaluationRecord[]): void {
  const identities = new Map<string, EffectiveEvaluationRecord[]>()
  for (const record of records) {
    const evaluation = record.evaluation
    if (evaluation.status !== 'available' || !record.benchmark) continue
    for (const metric of Object.keys(evaluation.metrics)) {
      const identity = JSON.stringify([
        evaluation.model,
        evaluation.reasoning_effort,
        evaluation.benchmark,
        evaluation.benchmark_profile,
        metric,
      ])
      const matches = identities.get(identity) ?? []
      matches.push(record)
      identities.set(identity, matches)
    }
  }
  for (const matches of identities.values()) {
    if (matches.length < 2) continue
    const issue = 'Duplicate available metric; resolve the conflicting records before applying.'
    for (const record of matches) {
      if (!record.issues.includes(issue)) record.issues.push(issue)
    }
  }
}

// Match the compiler's additive evidence policy. Editing still targets only
// evaluation.records; catalog evidence remains inherited, never copied into config.
export function buildEffectiveEvaluationGroups(
  config: ConfigData | null,
  models: NormalizedModel[],
  catalog: BuiltInModelCatalog | null,
): EffectiveEvaluationGroup[] {
  const benchmarks = new Map((catalog?.benchmarks ?? []).map((item) => [item.id, item]))
  const conflictingBenchmarks = new Set<string>()
  for (const benchmark of config?.evaluation?.benchmarks ?? []) {
    if (benchmarks.has(benchmark.id)) conflictingBenchmarks.add(benchmark.id)
    else benchmarks.set(benchmark.id, benchmark)
  }

  return models.map((model) => {
    const catalogId = model.catalog || model.name
    const records: EffectiveEvaluationRecord[] = []
    if (model.catalog) {
      for (const [index, evaluation] of (catalog?.evaluations ?? []).entries()) {
        if (evaluation.model !== catalogId) continue
        records.push({
          key: `built_in:${index}:${evaluation.id}`,
          origin: 'built_in',
          evaluation,
          benchmark: benchmarks.get(evaluation.benchmark),
          issues: [],
        })
      }
    }
    for (const [index, record] of (config?.evaluation?.records ?? []).entries()) {
      if (record.model === catalogId) {
        records.push(configuredEvidence(record, index, benchmarks.get(record.benchmark)))
      }
    }
    for (const record of records) {
      if (conflictingBenchmarks.has(record.evaluation.benchmark)) {
        record.issues.push(
          'Duplicate benchmark definition; resolve the conflicting definitions before applying.',
        )
      }
    }
    flagConflictingMetrics(records)
    return {
      modelName: model.name,
      catalogId,
      model,
      records,
      benchmarkCount: new Set(records.map((record) => record.evaluation.benchmark)).size,
      availableCount: records.filter((record) => record.evaluation.status === 'available').length,
      builtInCount: records.filter((record) => record.origin === 'built_in').length,
      configuredCount: records.filter((record) => record.origin === 'configured').length,
    }
  })
}
