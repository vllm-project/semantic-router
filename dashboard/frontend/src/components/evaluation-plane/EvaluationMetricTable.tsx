import { useDeferredValue, useEffect, useMemo, useState } from 'react'

import type { EvidenceLevel, EvaluationTrackId } from '../../types/evaluationPlane'
import type { EvaluationMetric } from '../../types/evaluationReport'
import { EvaluationActionButton } from './EvaluationPrimitives'
import EvaluationIssueDetails from './EvaluationIssueDetails'
import { evaluationMetricLabel } from './evaluationMetricPresentation'
import {
  evaluationResultScopeLabel,
  formatConfidenceInterval,
  formatDelta,
  formatMetric,
  isServerReducedMetric,
  metricDeltaTone,
} from './evaluationPresentation'
import { TRACK_PRESENTATION } from './evaluationTrackPresentation'
import styles from './EvaluationMetricTable.module.css'
import reportStyles from './EvaluationReportLayout.module.css'
import tableStyles from './EvaluationReportTable.module.css'

interface EvaluationMetricTableProps {
  metrics: EvaluationMetric[]
  caption?: string
  compact?: boolean
  controls?: boolean
  evidenceLevel?: EvidenceLevel
}

const METRICS_PAGE_SIZE = 20

function directionLabel(direction: EvaluationMetric['direction']): string {
  switch (direction) {
    case 'higher_is_better':
      return 'Higher is better'
    case 'lower_is_better':
      return 'Lower is better'
    case 'target':
      return 'Target range'
    default:
      return 'Diagnostic'
  }
}

function metricEvidenceLabel(
  evidenceLevel: EvidenceLevel | undefined,
  metricID: string,
): string | undefined {
  if (!evidenceLevel) return undefined
  return isServerReducedMetric(metricID)
    ? `Verified result · ${evaluationResultScopeLabel(evidenceLevel)}`
    : `Supporting diagnostic · ${evaluationResultScopeLabel(evidenceLevel)}`
}

function MetricValue({ metric }: { metric: EvaluationMetric }) {
  if (metric.value === null || !Number.isFinite(metric.value)) {
    return (
      <span className={styles.missingValue} title="This run did not produce this metric.">
        Not measured
      </span>
    )
  }
  return <>{formatMetric(metric)}</>
}

function useMetricTableModel(metrics: EvaluationMetric[]) {
  const [search, setSearch] = useState('')
  const [track, setTrack] = useState<EvaluationTrackId | 'all'>('all')
  const [page, setPage] = useState(1)
  const deferredSearch = useDeferredValue(search.trim().toLowerCase())
  const tracks = useMemo(
    () =>
      [
        ...new Set(metrics.map((metric) => metric.track_id).filter(Boolean)),
      ].sort() as EvaluationTrackId[],
    [metrics],
  )
  const filtered = useMemo(
    () =>
      metrics.filter((metric) => {
        if (track !== 'all' && metric.track_id !== track) return false
        if (!deferredSearch) return true
        return `${metric.id} ${metric.name} ${metric.track_id || ''} ${metric.unit}`
          .toLowerCase()
          .includes(deferredSearch)
      }),
    [deferredSearch, metrics, track],
  )
  const pages = Math.max(1, Math.ceil(filtered.length / METRICS_PAGE_SIZE))
  const safePage = Math.min(page, pages)
  const firstVisibleIndex = (safePage - 1) * METRICS_PAGE_SIZE
  const visible = filtered.slice(firstVisibleIndex, firstVisibleIndex + METRICS_PAGE_SIZE)

  useEffect(() => setPage(1), [deferredSearch, track])
  useEffect(() => {
    if (page > pages) setPage(pages)
  }, [page, pages])
  return {
    search,
    track,
    page: safePage,
    tracks,
    filtered,
    visible,
    pages,
    firstVisibleIndex,
    setSearch,
    setTrack,
    setPage,
  }
}

interface MetricToolbarProps {
  metricsCount: number
  filteredCount: number
  visibleCount: number
  firstVisibleIndex: number
  search: string
  track: EvaluationTrackId | 'all'
  tracks: EvaluationTrackId[]
  onSearchChange: (value: string) => void
  onTrackChange: (value: EvaluationTrackId | 'all') => void
}

function MetricToolbar({
  metricsCount,
  filteredCount,
  visibleCount,
  firstVisibleIndex,
  search,
  track,
  tracks,
  onSearchChange,
  onTrackChange,
}: MetricToolbarProps) {
  return (
    <div className={styles.metricToolbar}>
      <label>
        <span>Find a metric</span>
        <input
          type="search"
          value={search}
          placeholder="Metric name or evaluation area"
          onChange={(event) => onSearchChange(event.target.value)}
        />
      </label>
      {tracks.length > 1 ? (
        <label>
          <span>Evaluation area</span>
          <select
            value={track}
            onChange={(event) => onTrackChange(event.target.value as EvaluationTrackId | 'all')}
          >
            <option value="all">All areas</option>
            {tracks.map((trackID) => (
              <option key={trackID} value={trackID}>
                {TRACK_PRESENTATION[trackID].label}
              </option>
            ))}
          </select>
        </label>
      ) : null}
      <span className={styles.metricResultCount} aria-live="polite">
        {filteredCount
          ? `${firstVisibleIndex + 1}–${firstVisibleIndex + visibleCount} of ${filteredCount}`
          : `0 of ${metricsCount}`}
        {filteredCount !== metricsCount ? ` matching · ${metricsCount} total` : ''}
      </span>
    </div>
  )
}

function MetricRow({
  metric,
  compact,
  evidenceLevel,
}: {
  metric: EvaluationMetric
  compact: boolean
  evidenceLevel: EvidenceLevel | undefined
}) {
  const delta = formatDelta(metric)
  const interval = formatConfidenceInterval(metric)
  const deltaTone = metricDeltaTone(metric)
  const metricEvidence = metricEvidenceLabel(evidenceLevel, metric.id)
  return (
    <tr key={`${metric.track_id || 'all'}-${metric.id}`} data-metric-id={metric.id}>
      <th scope="row">
        <span className={styles.metricName}>{evaluationMetricLabel(metric)}</span>
        <span className={styles.metricIdentity}>
          {metric.track_id ? TRACK_PRESENTATION[metric.track_id].label : 'System'} ·{' '}
          {metricEvidence || 'Supporting diagnostic'}
        </span>
        <EvaluationIssueDetails
          className={styles.metricTechnicalDetails}
          issues={[
            { label: 'Metric ID', message: metric.id },
            { label: 'Reported metric name', message: metric.name },
            { label: 'Reported unit', message: metric.unit || 'No unit recorded' },
          ]}
        />
      </th>
      <td>
        <strong className={styles.metricValue}>
          <MetricValue metric={metric} />
        </strong>
        <span className={styles.metricDirection}>{directionLabel(metric.direction)}</span>
      </td>
      {!compact ? (
        <td>
          {metric.baseline_value !== null && typeof metric.baseline_value !== 'undefined' ? (
            <>
              <span>{formatMetric({ value: metric.baseline_value, unit: metric.unit })}</span>
              <strong className={styles[`delta_${deltaTone}`]}>{delta || 'No change'}</strong>
            </>
          ) : (
            <span className={styles.tableMuted}>No paired baseline</span>
          )}
        </td>
      ) : null}
      {!compact ? (
        <td>
          {interval ? (
            <span>{interval}</span>
          ) : (
            <span className={styles.tableMuted}>Not estimated</span>
          )}
        </td>
      ) : null}
      <td>
        {typeof metric.sample_count === 'number' ? (
          new Intl.NumberFormat().format(metric.sample_count)
        ) : (
          <span className={styles.tableMuted}>—</span>
        )}
      </td>
    </tr>
  )
}

function MetricResults({
  visible,
  caption,
  compact,
  evidenceLevel,
}: {
  visible: EvaluationMetric[]
  caption: string
  compact: boolean
  evidenceLevel: EvidenceLevel | undefined
}) {
  if (visible.length === 0) {
    return <p className={reportStyles.empty}>No metrics match the current filters.</p>
  }
  return (
    <div
      className={tableStyles.tableScroll}
      role="region"
      tabIndex={0}
      aria-label={`Scrollable ${caption}`}
    >
      <table
        className={`${tableStyles.table} ${styles.metricTable} ${compact ? styles.metricTableCompact : ''}`}
      >
        <caption>{caption}</caption>
        <thead>
          <tr>
            <th scope="col">Metric</th>
            <th scope="col">Value</th>
            {!compact ? <th scope="col">Baseline / delta</th> : null}
            {!compact ? <th scope="col">95% interval</th> : null}
            <th scope="col">Samples</th>
          </tr>
        </thead>
        <tbody>
          {visible.map((metric) => (
            <MetricRow
              key={`${metric.track_id || 'all'}-${metric.id}`}
              metric={metric}
              compact={compact}
              evidenceLevel={evidenceLevel}
            />
          ))}
        </tbody>
      </table>
    </div>
  )
}

function MetricPagination({
  page,
  pages,
  onPrevious,
  onNext,
}: {
  page: number
  pages: number
  onPrevious: () => void
  onNext: () => void
}) {
  return (
    <nav className={styles.metricPagination} aria-label="Metric table pages">
      <EvaluationActionButton type="button" compact disabled={page === 1} onClick={onPrevious}>
        Previous
      </EvaluationActionButton>
      <span>
        Page {page} of {pages}
      </span>
      <EvaluationActionButton type="button" compact disabled={page === pages} onClick={onNext}>
        Next
      </EvaluationActionButton>
    </nav>
  )
}

export default function EvaluationMetricTable({
  metrics,
  caption = 'Evaluation metrics',
  compact = false,
  controls = true,
  evidenceLevel,
}: EvaluationMetricTableProps) {
  const model = useMetricTableModel(metrics)

  if (metrics.length === 0) {
    return <p className={reportStyles.empty}>No metrics were produced for this result.</p>
  }

  return (
    <div className={styles.metricTableRegion}>
      {controls && metrics.length > 6 ? (
        <MetricToolbar
          metricsCount={metrics.length}
          filteredCount={model.filtered.length}
          visibleCount={model.visible.length}
          firstVisibleIndex={model.firstVisibleIndex}
          search={model.search}
          track={model.track}
          tracks={model.tracks}
          onSearchChange={model.setSearch}
          onTrackChange={model.setTrack}
        />
      ) : null}
      <MetricResults
        visible={model.visible}
        caption={caption}
        compact={compact}
        evidenceLevel={evidenceLevel}
      />
      {model.filtered.length > 0 && model.pages > 1 ? (
        <MetricPagination
          page={model.page}
          pages={model.pages}
          onPrevious={() => model.setPage((value) => Math.max(1, value - 1))}
          onNext={() => model.setPage((value) => Math.min(model.pages, value + 1))}
        />
      ) : null}
    </div>
  )
}
