import { useEffect, useMemo, useState } from 'react'

import type { BuiltInModelCatalog, CatalogBenchmark } from '../types/modelCatalog'
import {
  modelHubBenchmarkPoints,
  modelHubChartHues,
  modelHubDefaultBenchmark,
  paginateModelHubRows,
  type ModelHubBenchmarkPoint,
  type ModelHubBenchmarkSelection,
  type ModelHubRow,
} from './modelHubSupport'

const availableSelections = (
  catalog: BuiltInModelCatalog,
  modelIDs: Set<string>,
): ModelHubBenchmarkSelection[] => {
  const selections = new Map<string, ModelHubBenchmarkSelection>()
  catalog.evaluations.forEach((evaluation) => {
    if (evaluation.status !== 'available' || !modelIDs.has(evaluation.model)) return
    Object.keys(evaluation.metrics).forEach((metric) => {
      const selection = {
        benchmark: evaluation.benchmark,
        profile: evaluation.benchmark_profile,
        metric,
      }
      selections.set(`${selection.benchmark}\u0000${selection.profile}\u0000${metric}`, selection)
    })
  })
  return [...selections.values()]
}

const benchmarkOptionSets = (
  selections: ModelHubBenchmarkSelection[],
  selection: ModelHubBenchmarkSelection | null,
) => ({
  benchmarks: [...new Set(selections.map((item) => item.benchmark))],
  profiles: [
    ...new Set(
      selections
        .filter((item) => item.benchmark === selection?.benchmark)
        .map((item) => item.profile),
    ),
  ],
  metrics: [
    ...new Set(
      selections
        .filter(
          (item) => item.benchmark === selection?.benchmark && item.profile === selection?.profile,
        )
        .map((item) => item.metric),
    ),
  ],
})

const benchmarkChartDomain = (
  values: number[],
  metric: CatalogBenchmark['metrics'][number] | undefined,
): [number, number] => {
  if (metric?.unit === 'proportion' || metric?.unit === 'fraction') return metric.range
  const minimum = Math.min(...values)
  const maximum = Math.max(...values)
  if (!Number.isFinite(minimum) || !Number.isFinite(maximum)) return metric?.range ?? [0, 1]
  return minimum === maximum ? [minimum - 1, maximum + 1] : [minimum, maximum]
}

const useBenchmarkSelection = (
  catalog: BuiltInModelCatalog,
  selections: ModelHubBenchmarkSelection[],
) => {
  const fallback = useMemo(() => modelHubDefaultBenchmark(catalog), [catalog])
  const [selection, setSelection] = useState<ModelHubBenchmarkSelection | null>(fallback)

  useEffect(() => {
    if (
      selection &&
      selections.some(
        (candidate) =>
          candidate.benchmark === selection.benchmark &&
          candidate.profile === selection.profile &&
          candidate.metric === selection.metric,
      )
    )
      return
    setSelection(selections[0] ?? null)
  }, [selection, selections])

  return { selection, setSelection }
}

const useBenchmarkPagination = (
  points: ModelHubBenchmarkPoint[],
  selection: ModelHubBenchmarkSelection | null,
  rows: ModelHubRow[],
  compact: boolean,
) => {
  const [page, setPage] = useState(1)
  const [pageSize, setPageSize] = useState(() => (compact ? 12 : 24))

  useEffect(() => setPage(1), [selection, rows])
  useEffect(() => {
    setPageSize(compact ? 12 : 24)
    setPage(1)
  }, [compact])

  const pagination = useMemo(
    () => paginateModelHubRows(points, page, pageSize),
    [page, pageSize, points],
  )
  return { pagination, setPage, setPageSize }
}

export function useModelHubBenchmarkController(
  catalog: BuiltInModelCatalog,
  rows: ModelHubRow[],
  compact: boolean,
) {
  const modelIDs = useMemo(() => new Set(rows.map((row) => row.model.id)), [rows])
  const selections = useMemo(() => availableSelections(catalog, modelIDs), [catalog, modelIDs])
  const { selection, setSelection } = useBenchmarkSelection(catalog, selections)
  const options = useMemo(() => benchmarkOptionSets(selections, selection), [selection, selections])
  const points = useMemo(
    () => (selection ? modelHubBenchmarkPoints(catalog, selection, modelIDs) : []),
    [catalog, modelIDs, selection],
  )
  const chartHues = useMemo(
    () => modelHubChartHues(points.map((point) => point.model.id)),
    [points],
  )
  const benchmark = catalog.benchmarks.find((candidate) => candidate.id === selection?.benchmark)
  const metric = benchmark?.metrics.find((candidate) => candidate.id === selection?.metric)
  const [minimum, maximum] = benchmarkChartDomain(
    points.map((point) => point.value),
    metric,
  )
  const span = maximum - minimum || 1
  const paging = useBenchmarkPagination(points, selection, rows, compact)

  const changeBenchmark = (benchmarkID: string): void => {
    const next = selections.find((item) => item.benchmark === benchmarkID)
    if (next) setSelection(next)
  }
  const changeProfile = (profile: string): void => {
    const next = selections.find(
      (item) => item.benchmark === selection?.benchmark && item.profile === profile,
    )
    if (next) setSelection(next)
  }
  const changeMetric = (metricID: string): void => {
    if (selection) setSelection({ ...selection, metric: metricID })
  }

  return {
    selection,
    ...options,
    points,
    chartHues,
    benchmark,
    metric,
    minimum,
    span,
    ...paging,
    changeBenchmark,
    changeProfile,
    changeMetric,
  }
}

export type ModelHubBenchmarkController = ReturnType<typeof useModelHubBenchmarkController>
