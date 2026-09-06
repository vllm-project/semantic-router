import React from 'react'

import type {
  BenchmarkMetric,
  BenchmarkRow,
  CatalogBenchmark,
} from '../../data/modelHubCatalogTypes'
import { modelHubEvaluationConditionLabel } from '../../data/modelHubEvaluationLabel'
import {
  modelHubBenchmarkBarHeight,
  modelHubChartColorToken,
  type ModelHubChartColor,
} from '../../data/modelHubBenchmarkSupport'
import { CatalogMark } from './ModelHubMark'
import {
  EmptyState,
  formatMetric,
  readable,
  SelectControl,
  srOnlyClass,
} from './ModelHubPrimitives'
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
  const height = benchmarkColumnHeight(row.value, domain, metric.direction)
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
      <strong>{value}</strong>
      <span className={styles.columnTrack} aria-hidden="true"><i /></span>
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

export function ModelHubBenchmark({
  benchmark,
  metric,
  profile,
  benchmarks,
  profiles,
  metrics,
  rows,
  totalResults,
  query,
  publisher,
  publishers,
  domain,
  colors,
  chooseBenchmark,
  chooseProfile,
  chooseMetric,
  setQuery,
  setPublisher,
  selectModel,
}: {
  benchmark?: CatalogBenchmark
  metric?: BenchmarkMetric
  profile: string
  benchmarks: CatalogBenchmark[]
  profiles: Array<[string, string]>
  metrics: Array<[string, string]>
  rows: BenchmarkRow[]
  totalResults: number
  query: string
  publisher: string
  publishers: string[]
  domain: [number, number]
  colors: Map<string, ModelHubChartColor>
  chooseBenchmark: (id: string) => void
  chooseProfile: (id: string) => void
  chooseMetric: (id: string) => void
  setQuery: (query: string) => void
  setPublisher: (publisher: string) => void
  selectModel: (id: string) => void
}) {
  if (!benchmark || !metric) {
    return <EmptyState title="No benchmark evidence yet" body="Published results will appear here." />
  }

  return (
    <div className={styles.shell}>
      <div className={styles.controls}>
        <SelectControl label="Benchmark" value={benchmark.id} options={benchmarks.map(item => [item.id, item.display_name])} onChange={chooseBenchmark} />
        <SelectControl label="Profile" value={profile} options={profiles} onChange={chooseProfile} />
        <SelectControl label="Metric" value={metric.id} options={metrics} onChange={chooseMetric} />
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
        <label className={styles.publisherFilter}>
          <span>Creator</span>
          <select value={publisher} onChange={event => setPublisher(event.target.value)}>
            <option value="all">All creators</option>
            {publishers.map(item => <option value={item} key={item}>{item}</option>)}
          </select>
        </label>
      </div>
      <header className={styles.chartHeader}>
        <span>
          <strong>{benchmark.display_name}</strong>
          <small>
            {readable(metric.id)}
            {' '}
            ·
            {' '}
            {metric.direction === 'lower_is_better' ? 'Lower is better' : 'Higher is better'}
          </small>
        </span>
        <span>
          <b>{rows.length}</b>
          {' '}
          {rows.length === totalResults ? 'published results' : `of ${totalResults} results`}
          {benchmark.source ? <a href={benchmark.source} target="_blank" rel="noreferrer">Source ↗</a> : null}
        </span>
      </header>
      {rows.length
        ? (
            <div className={styles.viewport} tabIndex={0} aria-label="Benchmark comparison with all filtered results">
              <div className={styles.chart} role="list" style={{ '--bar-count': rows.length } as React.CSSProperties}>
                <span className={styles.grid} aria-hidden="true" />
                {rows.map(row => (
                  <span className={styles.columnItem} role="listitem" key={`${row.model.id}:${row.evaluation.reasoning_effort}`}>
                    <BenchmarkColumn row={row} metric={metric} domain={domain} color={colors.get(row.model.id) ?? { lightness: 0.62, chroma: 0.16, hue: 0 }} onSelect={() => selectModel(row.model.id)} />
                  </span>
                ))}
              </div>
            </div>
          )
        : (
            <EmptyState title="No matching measurements" body="Change the benchmark or model filters." />
          )}
    </div>
  )
}
