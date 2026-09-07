import { useMemo, useState } from 'react'

import type {
  BuiltInModelCatalog,
  CatalogBenchmark,
  CatalogBenchmarkMetric,
} from '../types/modelCatalog'
import {
  modelHubBenchmarkDomain,
  modelHubBenchmarkOverviewSelections,
  modelHubBenchmarkPoints,
  modelHubChartColors,
  type ModelHubBenchmarkPoint,
  type ModelHubBenchmarkSelection,
  type ModelHubChartColor,
  type ModelHubRow,
} from './modelHubSupport'

export interface ModelHubBenchmarkChartData {
  key: string
  selection: ModelHubBenchmarkSelection
  benchmark?: CatalogBenchmark
  metric?: CatalogBenchmarkMetric
  points: ModelHubBenchmarkPoint[]
  chartColors: Map<string, ModelHubChartColor>
  minimum: number
  span: number
}

export function useModelHubBenchmarkController(catalog: BuiltInModelCatalog, rows: ModelHubRow[]) {
  const [domain, setDomain] = useState('all')
  const modelIDs = useMemo(() => new Set(rows.map((row) => row.model.id)), [rows])
  const selections = useMemo(() => modelHubBenchmarkOverviewSelections(catalog), [catalog])
  const chartColors = useMemo(
    () =>
      modelHubChartColors(
        catalog.models.map((model) => model.id),
        catalog.models.map((model) => model.id),
      ),
    [catalog.models],
  )
  const allCharts = useMemo(
    () =>
      selections.flatMap<ModelHubBenchmarkChartData>((selection) => {
        const points = modelHubBenchmarkPoints(catalog, selection, modelIDs)
        if (!points.length) return []
        const benchmark = catalog.benchmarks.find(
          (candidate) => candidate.id === selection.benchmark,
        )
        const metric = benchmark?.metrics.find((candidate) => candidate.id === selection.metric)
        const [minimum, maximum] = metric
          ? modelHubBenchmarkDomain(
              points.map((point) => point.value),
              metric,
            )
          : [0, 1]
        return [
          {
            key: `${selection.benchmark}/${selection.profile}/${selection.metric}`,
            selection,
            benchmark,
            metric,
            points,
            chartColors,
            minimum,
            span: maximum - minimum || 1,
          },
        ]
      }),
    [catalog, chartColors, modelIDs, selections],
  )
  const domains = useMemo(() => {
    const counts = new Map<string, number>()
    allCharts.forEach((chart) => {
      const chartDomain = chart.benchmark?.domain
      if (!chartDomain) return
      counts.set(chartDomain, (counts.get(chartDomain) ?? 0) + 1)
    })
    return Array.from(counts, ([id, count]) => ({ id, count })).sort((left, right) =>
      left.id.localeCompare(right.id),
    )
  }, [allCharts])
  const activeDomain =
    domain === 'all' || domains.some((candidate) => candidate.id === domain) ? domain : 'all'
  const charts =
    activeDomain === 'all'
      ? allCharts
      : allCharts.filter((chart) => chart.benchmark?.domain === activeDomain)

  return {
    charts,
    chartCount: allCharts.length,
    domains,
    domain: activeDomain,
    setDomain,
  }
}

export type ModelHubBenchmarkController = ReturnType<typeof useModelHubBenchmarkController>
