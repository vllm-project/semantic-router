import React from 'react'

import type { BuiltInModelCatalog, CatalogBenchmark } from '../types/modelCatalog'
import type { ModelHubBenchmarkController } from './modelHubBenchmarkController'
import { ModelMark } from './ModelHubComponents'
import { OpenModelButton } from './ModelHubOpenModelButton'
import {
  modelHubEvaluationConditionLabel,
  readableModelHubValue as readable,
  type ModelHubBenchmarkPoint,
  type ModelHubRow,
} from './modelHubSupport'
import styles from './ModelHubPage.module.css'

const benchmarkLabel = (benchmark: CatalogBenchmark | undefined, fallback: string): string =>
  benchmark?.display_name ?? fallback

const formatMetric = (value: number, benchmark: CatalogBenchmark | undefined, metric: string) => {
  const definition = benchmark?.metrics.find((candidate) => candidate.id === metric)
  if (definition?.unit === 'fraction' || (definition?.range[1] === 1 && value <= 1)) {
    return `${(value * 100).toFixed(1)}%`
  }
  return Number.isInteger(value) ? value.toLocaleString() : value.toFixed(2)
}

export const ModelHubBenchmarkControls: React.FC<{
  catalog: BuiltInModelCatalog
  controller: ModelHubBenchmarkController
}> = ({ catalog, controller }) => (
  <div className={styles.benchmarkControls}>
    <label>
      <span>Benchmark</span>
      <select
        value={controller.selection?.benchmark ?? ''}
        onChange={(event) => controller.changeBenchmark(event.target.value)}
      >
        {controller.benchmarks.map((id) => (
          <option key={id} value={id}>
            {benchmarkLabel(
              catalog.benchmarks.find((item) => item.id === id),
              id,
            )}
          </option>
        ))}
      </select>
    </label>
    <label>
      <span>Profile</span>
      <select
        value={controller.selection?.profile ?? ''}
        onChange={(event) => controller.changeProfile(event.target.value)}
      >
        {controller.profiles.map((profile) => (
          <option key={profile} value={profile}>
            {readable(profile)}
          </option>
        ))}
      </select>
    </label>
    <label>
      <span>Metric</span>
      <select
        value={controller.selection?.metric ?? ''}
        onChange={(event) => controller.changeMetric(event.target.value)}
      >
        {controller.metrics.map((metricID) => (
          <option key={metricID} value={metricID}>
            {readable(metricID)}
          </option>
        ))}
      </select>
    </label>
  </div>
)

const BenchmarkRow: React.FC<{
  point: ModelHubBenchmarkPoint
  row: ModelHubRow
  selected: ModelHubRow | null
  select: (id: string) => void
  rank: number
  controller: ModelHubBenchmarkController
}> = ({ point, row, selected, select, rank, controller }) => {
  const conditionLabel = modelHubEvaluationConditionLabel(point.model, point.reasoningEffort)
  const rawPosition = (point.value - controller.minimum) / controller.span
  const performance =
    controller.metric?.direction === 'lower_is_better' ? 1 - rawPosition : rawPosition
  const width = Math.max(8, Math.min(100, 8 + performance * 92))
  const value = formatMetric(point.value, controller.benchmark, controller.selection?.metric ?? '')

  return (
    <OpenModelButton
      row={row}
      selected={selected}
      select={select}
      className={styles.benchmarkRow}
      ariaLabel={`Rank ${rank}, ${point.model.display_name}, ${conditionLabel}, ${value}`}
    >
      <span className={styles.chartRank} aria-label={`Rank ${rank}`}>
        {String(rank).padStart(2, '0')}
      </span>
      <span className={styles.chartIdentity}>
        <ModelMark model={point.model} />
        <span>
          <strong>{point.model.display_name}</strong>
          <small>{conditionLabel}</small>
        </span>
      </span>
      <span className={styles.chartTrack}>
        <i
          style={{
            width: `${width}%`,
            background: `hsl(${controller.chartHues.get(point.model.id) ?? 0} 62% 65%)`,
          }}
        />
      </span>
      <strong className={styles.chartValue}>{value}</strong>
    </OpenModelButton>
  )
}

export const ModelHubBenchmarkChart: React.FC<{
  rows: ModelHubRow[]
  selected: ModelHubRow | null
  select: (id: string) => void
  controller: ModelHubBenchmarkController
}> = ({ rows, selected, select, controller }) => (
  <div className={styles.benchmarkChart}>
    {controller.pagination.items.map((point, index) => {
      const row = rows.find((candidate) => candidate.model.id === point.model.id)
      if (!row) return null
      return (
        <BenchmarkRow
          key={`${point.model.id}/${point.reasoningEffort}`}
          point={point}
          row={row}
          selected={selected}
          select={select}
          rank={controller.pagination.start + index}
          controller={controller}
        />
      )
    })}
  </div>
)
