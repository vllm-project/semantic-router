import React from 'react'

import type {
  BenchmarkMetric,
  BenchmarkRow,
  CatalogBenchmark,
} from '../../data/modelHubCatalogTypes'
import { modelHubEvaluationConditionLabel } from '../../data/modelHubEvaluationLabel'
import {
  modelHubBenchmarkBarHeight,
  modelHubBenchmarkNormalizedValue,
  modelHubBenchmarkRawValueLabel,
  modelHubChartColorToken,
  type ModelHubChartColor,
} from '../../data/modelHubBenchmarkSupport'
import { CatalogMark } from './ModelHubMark'
import { EmptyState, formatMetric, readable, SelectControl, srOnlyClass } from './ModelHubPrimitives'
import styles from './modelHubBenchmark.module.css'

export const benchmarkColumnHeight = (
  value: number,
  domain: [number, number],
  direction: BenchmarkMetric['direction'],
): number => {
  const [minimum, maximum] = domain
  return modelHubBenchmarkBarHeight(value, minimum, maximum, direction)
}

function BenchmarkColumn({
  row,
  metric,
  domain,
  color,
  onSelect,
}: {
  row: BenchmarkRow
  metric: BenchmarkMetric
  domain: [number, number]
  color: ModelHubChartColor
  onSelect: () => void
}) {
  const condition = modelHubEvaluationConditionLabel(row.model, row.evaluation.reasoning_effort)
  const value = formatMetric(row.value, metric)
  const rawValue = modelHubBenchmarkRawValueLabel(row.value, metric)
  const height = benchmarkColumnHeight(
    modelHubBenchmarkNormalizedValue(row.value, metric),
    domain,
    metric.direction,
  )
  const token = modelHubChartColorToken(color)

  return (
    <button
      type="button"
      className={styles.column}
      onClick={onSelect}
      aria-label={`${row.model.display_name}, ${condition}, ${value}. Open evidence.`}
      style={
        {
          '--bar-height': `${height}%`,
          '--bar-color': token,
          '--bar-soft': `color-mix(in oklab, ${token} 14%, transparent)`,
        } as React.CSSProperties
      }
    >
      <strong title={rawValue ? `Raw: ${rawValue}` : undefined}>{value}</strong>
      <span className={styles.columnTrack} aria-hidden="true">
        <i />
      </span>
      <span className={styles.columnIdentity}>
        <CatalogMark presentation={row.model.presentation} />
        <span>
          <b title={row.model.display_name}>{row.model.display_name}</b>
          <small>{condition}</small>
        </span>
      </span>
    </button>
  )
}

interface BenchmarkChart {
  key: string
  benchmark: CatalogBenchmark
  metric: BenchmarkMetric
  profile: string
  rows: BenchmarkRow[]
  totalResults: number
  domain: [number, number]
  colors: Map<string, ModelHubChartColor>
}

function BenchmarkPanel({
  chart,
  selectModel,
}: {
  chart: BenchmarkChart
  selectModel: (id: string) => void
}) {
  const profile = chart.benchmark.profiles.find(item => item.id === chart.profile)

  return (
    <article className={styles.panel}>
      <header className={styles.chartHeader}>
        <span>
          <strong>{chart.benchmark.display_name}</strong>
          <small>{profile?.description ?? readable(chart.profile)}</small>
        </span>
        {chart.benchmark.source
          ? (
              <a href={chart.benchmark.source} target="_blank" rel="noreferrer">
                Source ↗
              </a>
            )
          : null}
      </header>
      <div className={styles.chartMeta}>
        <span>
          {readable(chart.metric.id)}
          {' '}
          ·
          {' '}
          {chart.metric.direction === 'lower_is_better' ? 'Lower is better' : 'Higher is better'}
        </span>
        <span>
          <b>{chart.rows.length}</b>
          {' '}
          {chart.rows.length === chart.totalResults ? 'results' : `of ${chart.totalResults}`}
        </span>
      </div>
      <div
        className={styles.viewport}
        tabIndex={0}
        aria-label={`${chart.benchmark.display_name}. Benchmark comparison with all filtered results`}
      >
        <div
          className={styles.chart}
          role="list"
          style={{ '--bar-count': chart.rows.length } as React.CSSProperties}
        >
          <span className={styles.grid} aria-hidden="true" />
          {chart.rows.map(row => (
            <span
              className={styles.columnItem}
              role="listitem"
              key={`${row.model.id}:${row.evaluation.reasoning_effort}`}
            >
              <BenchmarkColumn
                row={row}
                metric={chart.metric}
                domain={chart.domain}
                color={
                  chart.colors.get(row.model.id) ?? {
                    lightness: 0.62,
                    chroma: 0.16,
                    hue: 0,
                  }
                }
                onSelect={() => selectModel(row.model.id)}
              />
            </span>
          ))}
        </div>
      </div>
    </article>
  )
}

export function ModelHubBenchmark({
  charts,
  chartCount,
  filters,
  filter,
  setFilter,
  query,
  publisher,
  publishers,
  setQuery,
  setPublisher,
  selectModel,
}: {
  charts: BenchmarkChart[]
  chartCount: number
  filters: Array<{ id: string, label: string, count: number }>
  filter: string
  setFilter: (filter: string) => void
  query: string
  publisher: string
  publishers: string[]
  setQuery: (query: string) => void
  setPublisher: (publisher: string) => void
  selectModel: (id: string) => void
}) {
  if (!chartCount) {
    return (
      <EmptyState title="No benchmark evidence yet" body="Published results will appear here." />
    )
  }

  return (
    <div className={styles.shell}>
      <div className={styles.toolbar}>
        <div className={styles.tags} role="group" aria-label="Filter benchmarks">
          <button type="button" aria-pressed={filter === 'all'} onClick={() => setFilter('all')}>
            All
            {' '}
            <span>{chartCount}</span>
          </button>
          {filters.map(item => (
            <button
              type="button"
              key={item.id}
              aria-pressed={filter === item.id}
              onClick={() => setFilter(item.id)}
            >
              {readable(item.label)}
              {' '}
              <span>{item.count}</span>
            </button>
          ))}
        </div>
      </div>
      <div className={styles.filters}>
        <label className={styles.search}>
          <span className={srOnlyClass}>Filter benchmark models</span>
          <svg viewBox="0 0 20 20" aria-hidden="true">
            <circle cx="8.5" cy="8.5" r="5.5" />
            <path d="m12.5 12.5 4 4" />
          </svg>
          <input
            type="search"
            value={query}
            onChange={event => setQuery(event.target.value)}
            placeholder="Filter models"
          />
        </label>
        <SelectControl
          label="Creator"
          value={publisher}
          options={[
            ['all', 'All creators'],
            ...publishers.map(item => [item, item] as [string, string]),
          ]}
          onChange={setPublisher}
        />
      </div>
      {charts.length
        ? (
            <div className={styles.gallery}>
              {charts.map(chart => (
                <BenchmarkPanel chart={chart} selectModel={selectModel} key={chart.key} />
              ))}
            </div>
          )
        : (
            <EmptyState
              title="No matching measurements"
              body="Change the benchmark or model filters."
            />
          )}
    </div>
  )
}
