import { readFileSync } from 'node:fs'

import { describe, expect, it } from 'vitest'

import {
  decodeMetricAnalysisSubjectID,
  DYNAMIC_METRIC_ANALYSIS_FAMILY_IDS,
  encodeMetricAnalysisSubjectID,
  METRIC_ANALYSIS_CATALOG_SOURCE,
  resolveMetricAnalysisCatalog,
  STATIC_METRIC_ANALYSIS_IDS,
  validateMetricAnalysisCatalogSource,
} from './metricAnalysisCatalog'

interface CatalogFixture {
  identifier_encoding: { vectors: Array<{ raw: string; encoded: string }> }
  analysis_templates: Array<Record<string, unknown> & { id: string }>
  static_metrics: Array<{ id: string; analysis_ref: string }>
  dynamic_families: Array<{
    id: string
    literal_prefix: string
    pattern: string
    examples: Array<{
      metric_id: string
      captures: Record<string, string>
      analysis_ref: string
    }>
  }>
}

const catalog = JSON.parse(METRIC_ANALYSIS_CATALOG_SOURCE) as CatalogFixture

function compareCanonicalCatalogIDs(
  left: Readonly<{ id: string }>,
  right: Readonly<{ id: string }>,
): number {
  if (left.id < right.id) return -1
  if (left.id > right.id) return 1
  return 0
}

describe('canonical metric analysis catalog', () => {
  it('keeps generated Go and TypeScript mirrors byte-identical to Python', () => {
    const typescriptResource = readFileSync(
      new URL('./metric_analysis_catalog.v1.json', import.meta.url),
      'utf8',
    )
    const pythonResource = readFileSync(
      new URL(
        '../../../../src/vllm-sr/cli/evaluation/golden/metric_analysis_catalog.v1.json',
        import.meta.url,
      ),
      'utf8',
    )
    const goResource = readFileSync(
      new URL('../../../backend/evaluationplane/metric_analysis_catalog.v1.json', import.meta.url),
      'utf8',
    )

    expect(typescriptResource).toBe(pythonResource)
    expect(typescriptResource).toBe(goResource)
    expect(JSON.parse(METRIC_ANALYSIS_CATALOG_SOURCE)).toEqual(JSON.parse(typescriptResource))
  })

  it('resolves all 136 exact ids and all six typed dynamic families', () => {
    expect(STATIC_METRIC_ANALYSIS_IDS).toHaveLength(136)
    expect(DYNAMIC_METRIC_ANALYSIS_FAMILY_IDS).toHaveLength(6)
    expect(STATIC_METRIC_ANALYSIS_IDS).toEqual([...STATIC_METRIC_ANALYSIS_IDS].sort())
    expect(DYNAMIC_METRIC_ANALYSIS_FAMILY_IDS).toEqual(
      [...DYNAMIC_METRIC_ANALYSIS_FAMILY_IDS].sort(),
    )

    for (const metricID of STATIC_METRIC_ANALYSIS_IDS) {
      const match = resolveMetricAnalysisCatalog(metricID)
      expect(match.metric_id).toBe(metricID)
      expect(match.family_id).toBeUndefined()
    }
    for (const family of catalog.dynamic_families) {
      for (const example of family.examples) {
        const match = resolveMetricAnalysisCatalog(example.metric_id)
        expect(match.family_id).toBe(family.id)
        expect(match.captures).toEqual(example.captures)
        expect(match.specification.id).toBe(example.analysis_ref)
      }
    }
  })

  it('uses a canonical one-segment codec, including Router colon keys', () => {
    for (const vector of catalog.identifier_encoding.vectors) {
      expect(encodeMetricAnalysisSubjectID(vector.raw)).toBe(vector.encoded)
      expect(decodeMetricAnalysisSubjectID(vector.encoded)).toBe(vector.raw)
      expect(vector.encoded).not.toContain('.')
      expect(vector.encoded).not.toContain(':')
    }
    expect(encodeMetricAnalysisSubjectID('domain:reasoning')).toBe('u-ZG9tYWluOnJlYXNvbmluZw')
    expect(encodeMetricAnalysisSubjectID('classifier:risk:RISKY')).toBe(
      'u-Y2xhc3NpZmllcjpyaXNrOlJJU0tZ',
    )
  })

  it('fails closed for unknown, malformed, and ambiguous metric ids', () => {
    for (const metricID of [
      'routing.made_up_accuracy',
      'model_pool.arm.u-abc.quality',
      'capacity.level.0.success_rate',
      'routing_recipe.e2.feasible_oracle_recall_at_65',
    ]) {
      expect(() => resolveMetricAnalysisCatalog(metricID)).toThrow()
    }

    const ambiguous = JSON.parse(METRIC_ANALYSIS_CATALOG_SOURCE) as CatalogFixture
    ambiguous.dynamic_families[1].literal_prefix = ambiguous.dynamic_families[0].literal_prefix
    ambiguous.dynamic_families[1].pattern = ambiguous.dynamic_families[0].pattern
    expect(() => validateMetricAnalysisCatalogSource(JSON.stringify(ambiguous))).toThrow(
      /prefixes overlap/,
    )
  })

  it('accepts sorted referenced extensions without production cardinality gates', () => {
    const extensible = JSON.parse(METRIC_ANALYSIS_CATALOG_SOURCE) as CatalogFixture
    const source = extensible.analysis_templates.find((item) => item.id === 'routing.case.ratio')
    expect(source).toBeDefined()
    extensible.analysis_templates.push({
      ...source!,
      id: 'routing.catalog-extensibility-probe',
    })
    extensible.analysis_templates.sort(compareCanonicalCatalogIDs)
    extensible.static_metrics.push({
      id: 'routing.catalog_extensibility_probe',
      analysis_ref: 'routing.catalog-extensibility-probe',
    })
    extensible.static_metrics.sort(compareCanonicalCatalogIDs)

    expect(() => validateMetricAnalysisCatalogSource(JSON.stringify(extensible))).not.toThrow()
  })

  it('rejects unreferenced templates and removed root baggage', () => {
    const orphaned = JSON.parse(METRIC_ANALYSIS_CATALOG_SOURCE) as CatalogFixture
    orphaned.analysis_templates.push({
      ...orphaned.analysis_templates[0],
      id: 'agentic.unreferenced-probe',
    })
    orphaned.analysis_templates.sort(compareCanonicalCatalogIDs)
    expect(() => validateMetricAnalysisCatalogSource(JSON.stringify(orphaned))).toThrow(
      /referenced exhaustively/,
    )

    const baggage = JSON.parse(METRIC_ANALYSIS_CATALOG_SOURCE) as Record<string, unknown>
    baggage.legacy_metric_inventory = []
    expect(() => validateMetricAnalysisCatalogSource(JSON.stringify(baggage))).toThrow(
      /root fields are invalid/,
    )
  })

  it('returns the exact estimator contract instead of inferring from the metric name', () => {
    expect(resolveMetricAnalysisCatalog('routing.accuracy').specification).toMatchObject({
      estimator_id: 'deterministic-routing-case-observed-ratio',
      estimator_version: 'v1',
      analysis_unit: 'route_case',
      cluster_unit: 'case',
      weighting: 'uniform_case',
      missingness: 'fail_closed',
      exclusion_policy: 'exclude_unavailable_evidence',
    })
    expect(
      resolveMetricAnalysisCatalog('capacity.level.16.error_rate_upper_bound').specification,
    ).toMatchObject({
      estimator_id: 'capacity-level-worst-cluster-one-sided-wilson-upper',
      analysis_unit: 'measurement_cluster',
      cluster_unit: 'measurement_cluster',
      weighting: 'worst_cluster',
    })
    for (const metricID of [
      'capacity.error_rate',
      'capacity.success_rate',
      'capacity.level.16.error_rate',
      'capacity.level.16.success_rate',
    ]) {
      expect(resolveMetricAnalysisCatalog(metricID).specification).toMatchObject({
        analysis_unit: 'measurement_cluster',
        cluster_unit: 'measurement_cluster',
        weighting: 'uniform_cluster',
      })
    }
  })
})
