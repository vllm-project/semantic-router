import React from 'react'

import type { BuiltInModelCatalog, CatalogBenchmark } from '../types/modelCatalog'
import type { ModelHubBenchmarkController } from './modelHubBenchmarkController'
import { ModelMark } from './ModelHubComponents'
import { OpenModelButton } from './ModelHubOpenModelButton'
import {
  modelHubBenchmarkBarHeight,
  modelHubChartColorToken,
  modelHubEvaluationConditionLabel,
  readableModelHubValue as readable,
  type ModelHubBenchmarkPoint,
  type ModelHubRow,
} from './modelHubSupport'
import styles from './ModelHubViews.module.css'

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

const BenchmarkColumn: React.FC<{
  point: ModelHubBenchmarkPoint
  row: ModelHubRow
  selected: ModelHubRow | null
  select: (id: string) => void
  controller: ModelHubBenchmarkController
}> = ({ point, row, selected, select, controller }) => {
  const conditionLabel = modelHubEvaluationConditionLabel(point.model, point.reasoningEffort)
  const height = modelHubBenchmarkBarHeight(
    point.value,
    controller.minimum,
    controller.minimum + controller.span,
    controller.metric?.direction ?? 'higher_is_better',
  )
  const value = formatMetric(point.value, controller.benchmark, controller.selection?.metric ?? '')
  const color = controller.chartColors.get(point.model.id) ?? {
    lightness: 0.62,
    chroma: 0.16,
    hue: 0,
  }
  const colorToken = modelHubChartColorToken(color)

  return (
    <OpenModelButton
      row={row}
      selected={selected}
      select={select}
      className={styles.benchmarkColumn}
      ariaLabel={`${point.model.display_name}, ${conditionLabel}, ${value}. Open evidence.`}
      style={
        {
          '--column-height': `${height}%`,
          '--column-color': colorToken,
          '--column-soft': `color-mix(in oklab, ${colorToken} 16%, transparent)`,
        } as React.CSSProperties
      }
    >
      <strong className={styles.columnValue}>{value}</strong>
      <span className={styles.columnTrack} aria-hidden="true">
        <i />
      </span>
      <span className={styles.columnIdentity}>
        <ModelMark model={point.model} />
        <span>
          <strong title={point.model.display_name}>{point.model.display_name}</strong>
          <small>{conditionLabel}</small>
        </span>
      </span>
    </OpenModelButton>
  )
}

export const ModelHubBenchmarkChart: React.FC<{
  rows: ModelHubRow[]
  selected: ModelHubRow | null
  select: (id: string) => void
  controller: ModelHubBenchmarkController
}> = ({ rows, selected, select, controller }) => (
  <div
    className={styles.benchmarkViewport}
    tabIndex={0}
    aria-label="Benchmark comparison with all filtered results"
  >
    <div
      className={styles.benchmarkChart}
      role="list"
      style={{ '--column-count': controller.points.length } as React.CSSProperties}
    >
      <span className={styles.chartGrid} aria-hidden="true" />
      {controller.points.map((point) => {
        const row = rows.find((candidate) => candidate.model.id === point.model.id)
        if (!row) return null
        return (
          <span
            role="listitem"
            className={styles.benchmarkColumnItem}
            key={`${point.model.id}/${point.reasoningEffort}`}
          >
            <BenchmarkColumn
              point={point}
              row={row}
              selected={selected}
              select={select}
              controller={controller}
            />
          </span>
        )
      })}
    </div>
  </div>
)
