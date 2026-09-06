import { useEffect, useMemo, useState } from 'react'

import type { BuiltInModelCatalog } from '../types/modelCatalog'
import {
  modelHubBenchmarkDomain,
  modelHubBenchmarkPoints,
  modelHubBenchmarkSelections,
  modelHubChartColors,
  modelHubDefaultBenchmark,
  type ModelHubBenchmarkSelection,
  type ModelHubRow,
} from './modelHubSupport'

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

export function useModelHubBenchmarkController(catalog: BuiltInModelCatalog, rows: ModelHubRow[]) {
  const modelIDs = useMemo(() => new Set(rows.map((row) => row.model.id)), [rows])
  const selections = useMemo(() => modelHubBenchmarkSelections(catalog), [catalog])
  const { selection, setSelection } = useBenchmarkSelection(catalog, selections)
  const options = useMemo(() => benchmarkOptionSets(selections, selection), [selection, selections])
  const points = useMemo(
    () => (selection ? modelHubBenchmarkPoints(catalog, selection, modelIDs) : []),
    [catalog, modelIDs, selection],
  )
  const chartColors = useMemo(
    () =>
      modelHubChartColors(
        points.map((point) => point.model.id),
        catalog.models.map((model) => model.id),
      ),
    [catalog.models, points],
  )
  const benchmark = catalog.benchmarks.find((candidate) => candidate.id === selection?.benchmark)
  const metric = benchmark?.metrics.find((candidate) => candidate.id === selection?.metric)
  const [minimum, maximum] = metric
    ? modelHubBenchmarkDomain(
        points.map((point) => point.value),
        metric,
      )
    : [0, 1]
  const span = maximum - minimum || 1

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
    chartColors,
    benchmark,
    metric,
    minimum,
    span,
    changeBenchmark,
    changeProfile,
    changeMetric,
  }
}

export type ModelHubBenchmarkController = ReturnType<typeof useModelHubBenchmarkController>
