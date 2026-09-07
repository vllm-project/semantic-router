import React from 'react'

import type { ModelHubBenchmarkChartData } from './modelHubBenchmarkController'
import {
  modelHubBenchmarkNormalizedValue,
  modelHubBenchmarkRawValueLabel,
  modelHubBenchmarkValueLabel,
} from './modelHubBenchmarkNormalization'
import { ModelMark } from './ModelHubComponents'
import { OpenModelButton } from './ModelHubOpenModelButton'
import {
  modelHubBenchmarkBarHeight,
  modelHubChartColorToken,
  modelHubEvaluationConditionLabel,
  type ModelHubBenchmarkPoint,
  type ModelHubRow,
} from './modelHubSupport'
import styles from './ModelHubViews.module.css'

const BenchmarkColumn: React.FC<{
  point: ModelHubBenchmarkPoint
  row: ModelHubRow
  openModel: (id: string) => void
  chart: ModelHubBenchmarkChartData
}> = ({ point, row, openModel, chart }) => {
  const conditionLabel = modelHubEvaluationConditionLabel(point.model, point.reasoningEffort)
  const metric = chart.metric
  const chartValue = metric ? modelHubBenchmarkNormalizedValue(point.value, metric) : point.value
  const height = modelHubBenchmarkBarHeight(
    chartValue,
    chart.minimum,
    chart.minimum + chart.span,
    chart.metric?.direction ?? 'higher_is_better',
  )
  const value = metric
    ? modelHubBenchmarkValueLabel(point.value, metric)
    : Number.isInteger(point.value)
      ? point.value.toLocaleString()
      : point.value.toFixed(2)
  const rawValue = metric ? modelHubBenchmarkRawValueLabel(point.value, metric) : undefined
  const color = chart.chartColors.get(point.model.id) ?? {
    lightness: 0.62,
    chroma: 0.16,
    hue: 0,
  }
  const colorToken = modelHubChartColorToken(color)

  return (
    <OpenModelButton
      row={row}
      select={openModel}
      className={styles.benchmarkColumn}
      ariaLabel={`${point.model.display_name}, ${conditionLabel}, ${value}. Open model details.`}
      style={
        {
          '--column-height': `${height}%`,
          '--column-color': colorToken,
          '--column-soft': `color-mix(in oklab, ${colorToken} 16%, transparent)`,
        } as React.CSSProperties
      }
    >
      <strong className={styles.columnValue} title={rawValue ? `Raw: ${rawValue}` : undefined}>
        {value}
      </strong>
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
  openModel: (id: string) => void
  chart: ModelHubBenchmarkChartData
}> = ({ rows, openModel, chart }) => (
  <div
    className={styles.benchmarkViewport}
    tabIndex={0}
    aria-label={`${chart.benchmark?.display_name ?? chart.selection.benchmark} comparison with all filtered results`}
  >
    <div
      className={styles.benchmarkChart}
      role="list"
      style={{ '--column-count': chart.points.length } as React.CSSProperties}
    >
      <span className={styles.chartGrid} aria-hidden="true" />
      {chart.points.map((point) => {
        const row = rows.find((candidate) => candidate.model.id === point.model.id)
        if (!row) return null
        return (
          <span
            role="listitem"
            className={styles.benchmarkColumnItem}
            key={`${point.model.id}/${point.reasoningEffort}`}
          >
            <BenchmarkColumn point={point} row={row} openModel={openModel} chart={chart} />
          </span>
        )
      })}
    </div>
  </div>
)
