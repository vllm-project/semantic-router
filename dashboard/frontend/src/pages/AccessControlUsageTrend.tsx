import { useMemo, useState, type KeyboardEvent, type PointerEvent } from 'react'

import type { UsagePoint, UsageSummary } from '../utils/inferenceAccessApi'
import styles from './AccessControlPage.module.css'

type Metric = 'tokens' | 'requests' | 'latency'

interface Props {
  points: UsagePoint[]
  metric: Metric
  granularity: UsageSummary['granularity']
}

const exactNumber = new Intl.NumberFormat('en-US')
const compactNumber = new Intl.NumberFormat('en-US', {
  notation: 'compact',
  maximumFractionDigits: 1,
})

const pointValue = (point: UsagePoint, metric: Metric) =>
  metric === 'tokens'
    ? point.totalTokens
    : metric === 'requests'
      ? point.requests
      : point.averageLatencyMs

const niceAxis = (maximum: number) => {
  if (maximum <= 4) return { maximum: 4, ticks: [0, 1, 2, 3, 4] }
  const roughStep = maximum / 4
  const magnitude = 10 ** Math.floor(Math.log10(roughStep))
  const normalized = roughStep / magnitude
  const step = (normalized <= 1 ? 1 : normalized <= 2 ? 2 : normalized <= 5 ? 5 : 10) * magnitude
  const axisMaximum = step * Math.ceil(maximum / step)
  const count = Math.min(5, Math.max(2, Math.ceil(axisMaximum / step) + 1))
  const adjustedStep = axisMaximum / (count - 1)
  return {
    maximum: axisMaximum,
    ticks: Array.from({ length: count }, (_, index) => Math.round(adjustedStep * index)),
  }
}

const axisValue = (value: number, metric: Metric) =>
  metric === 'latency' ? `${compactNumber.format(value)} ms` : compactNumber.format(value)

const bucketLabel = (bucket: string, granularity: Props['granularity'], long = false) => {
  const value = new Date(bucket)
  if (long) {
    return new Intl.DateTimeFormat('en-US', {
      month: 'short',
      day: 'numeric',
      hour: granularity === 'day' ? undefined : 'numeric',
      minute: granularity === 'minute' ? '2-digit' : undefined,
    }).format(value)
  }
  return new Intl.DateTimeFormat('en-US', {
    month: granularity === 'day' ? 'short' : undefined,
    day: granularity === 'day' ? 'numeric' : undefined,
    weekday: granularity === 'hour' ? 'short' : undefined,
    hour: granularity === 'day' ? undefined : 'numeric',
    minute: granularity === 'minute' ? '2-digit' : undefined,
  }).format(value)
}

export default function AccessControlUsageTrend({ points, metric, granularity }: Props) {
  const [activeIndex, setActiveIndex] = useState<number | null>(null)
  const chart = useMemo(() => {
    const values = points.map((point) => pointValue(point, metric))
    const axis = niceAxis(Math.max(...values, 0))
    const width = 1000
    const height = 250
    const coordinates = points.map((point, index) => ({
      point,
      value: values[index],
      x: points.length === 1 ? width / 2 : (index / (points.length - 1)) * width,
      y: height - (values[index] / axis.maximum) * height,
    }))
    const line = coordinates
      .map((point, index) => `${index ? 'L' : 'M'}${point.x.toFixed(1)},${point.y.toFixed(1)}`)
      .join(' ')
    const area = coordinates.length
      ? `${line} L${coordinates[coordinates.length - 1].x},${height} L${coordinates[0].x},${height} Z`
      : ''
    return { axis, width, height, coordinates, line, area }
  }, [metric, points])

  if (!points.length) {
    return (
      <div className={styles.usageEmpty}>
        <strong>No usage yet</strong>
        <span>Managed requests will appear here.</span>
      </div>
    )
  }

  const selected = activeIndex === null ? null : chart.coordinates[activeIndex]
  const labelIndexes = Array.from(
    new Set(
      Array.from({ length: Math.min(6, points.length) }, (_, index) =>
        Math.round((index / Math.max(Math.min(6, points.length) - 1, 1)) * (points.length - 1)),
      ),
    ),
  )
  const selectFromPointer = (event: PointerEvent<HTMLDivElement>) => {
    const bounds = event.currentTarget.getBoundingClientRect()
    const ratio = Math.min(1, Math.max(0, (event.clientX - bounds.left) / bounds.width))
    setActiveIndex(Math.round(ratio * (points.length - 1)))
  }
  const moveSelection = (event: KeyboardEvent<HTMLDivElement>) => {
    if (event.key !== 'ArrowLeft' && event.key !== 'ArrowRight') return
    event.preventDefault()
    const direction = event.key === 'ArrowRight' ? 1 : -1
    setActiveIndex((current) =>
      Math.min(points.length - 1, Math.max(0, (current ?? points.length - 1) + direction)),
    )
  }

  return (
    <div className={styles.usageChartWrap}>
      <div className={styles.usageChartBody}>
        <div className={styles.usageYAxis} aria-hidden="true">
          {[...chart.axis.ticks].reverse().map((tick) => (
            <span key={tick}>{axisValue(tick, metric)}</span>
          ))}
        </div>
        <div
          className={styles.usageChartCanvas}
          role="img"
          aria-label={`${metric} over time. Use the arrow keys to inspect each point.`}
          tabIndex={0}
          onPointerMove={selectFromPointer}
          onPointerLeave={() => setActiveIndex(null)}
          onFocus={() => setActiveIndex((current) => current ?? points.length - 1)}
          onBlur={() => setActiveIndex(null)}
          onKeyDown={moveSelection}
        >
          <svg viewBox={`0 0 ${chart.width} ${chart.height}`} preserveAspectRatio="none">
            {chart.axis.ticks.map((tick) => {
              const y = chart.height - (tick / chart.axis.maximum) * chart.height
              return (
                <line
                  key={tick}
                  x1="0"
                  x2={chart.width}
                  y1={y}
                  y2={y}
                  className={styles.gridLine}
                />
              )
            })}
            <defs>
              <linearGradient id={`usage-${metric}`} x1="0" y1="0" x2="0" y2="1">
                <stop offset="0" stopColor="#70a5ff" stopOpacity=".32" />
                <stop offset="1" stopColor="#70a5ff" stopOpacity="0" />
              </linearGradient>
            </defs>
            <path d={chart.area} fill={`url(#usage-${metric})`} />
            <path d={chart.line} className={styles.chartLine} />
            {points.length <= 120
              ? chart.coordinates.map((item) => (
                  <circle
                    key={item.point.bucket}
                    cx={item.x}
                    cy={item.y}
                    r="3"
                    className={styles.chartPoint}
                  />
                ))
              : null}
            {selected ? (
              <>
                <line
                  x1={selected.x}
                  x2={selected.x}
                  y1="0"
                  y2={chart.height}
                  className={styles.chartCrosshair}
                />
                <circle cx={selected.x} cy={selected.y} r="5" className={styles.chartPointActive} />
              </>
            ) : null}
          </svg>
          {selected ? (
            <div
              className={styles.usageChartTooltip}
              style={{ left: `${Math.min(90, Math.max(10, (selected.x / chart.width) * 100))}%` }}
            >
              <span>{bucketLabel(selected.point.bucket, granularity, true)}</span>
              <strong>
                {exactNumber.format(selected.value)} {metric === 'latency' ? 'ms' : metric}
              </strong>
              <small>
                {metric === 'latency'
                  ? `${exactNumber.format(selected.point.p95LatencyMs)} ms P95 · ${exactNumber.format(selected.point.averageTtftMs)} ms first token`
                  : `${exactNumber.format(selected.point.requests)} requests · ${exactNumber.format(selected.point.totalTokens)} tokens`}
              </small>
            </div>
          ) : null}
        </div>
      </div>
      <div className={styles.chartLabels}>
        {labelIndexes.map((index) => (
          <span key={points[index].bucket}>{bucketLabel(points[index].bucket, granularity)}</span>
        ))}
      </div>
    </div>
  )
}
