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

export interface ModelHubBenchmarkFilter {
  id: string
  label: string
  count: number
}

export function useModelHubBenchmarkController(catalog: BuiltInModelCatalog, rows: ModelHubRow[]) {
  const [filter, setFilter] = useState('all')
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
  const filters = useMemo(() => {
    const tagCounts = new Map<string, number>()
    const domainCounts = new Map<string, number>()
    allCharts.forEach((chart) => {
      const chartDomain = chart.benchmark?.domain
      if (chartDomain) {
        domainCounts.set(chartDomain, (domainCounts.get(chartDomain) ?? 0) + 1)
      }
      chart.benchmark?.tags?.forEach((tag) => {
        tagCounts.set(tag, (tagCounts.get(tag) ?? 0) + 1)
      })
    })
    const tags = Array.from(tagCounts, ([tag, count]) => ({
      id: `tag:${tag}`,
      label: tag === 'core' ? 'Core' : tag,
      count,
    })).sort((left, right) => {
      if (left.id === 'tag:core') return -1
      if (right.id === 'tag:core') return 1
      return left.label.localeCompare(right.label)
    })
    const domains = Array.from(domainCounts, ([domain, count]) => ({
      id: `domain:${domain}`,
      label: domain,
      count,
    })).sort((left, right) => left.label.localeCompare(right.label))
    return [...tags, ...domains]
  }, [allCharts])
  const activeFilter =
    filter === 'all' || filters.some((candidate) => candidate.id === filter) ? filter : 'all'
  const charts =
    activeFilter === 'all'
      ? allCharts
      : activeFilter.startsWith('tag:')
        ? allCharts.filter((chart) => chart.benchmark?.tags?.includes(activeFilter.slice(4)))
        : allCharts.filter((chart) => chart.benchmark?.domain === activeFilter.slice(7))

  return {
    charts,
    chartCount: allCharts.length,
    filters,
    filter: activeFilter,
    setFilter,
  }
}

export type ModelHubBenchmarkController = ReturnType<typeof useModelHubBenchmarkController>
