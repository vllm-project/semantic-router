import React from 'react'

import type { BenchmarkMetric, CatalogModel } from '../../data/modelHubCatalogTypes'
import styles from './modelHubShared.module.css'

export const readable = (value: string) => value.replace(/_/g, ' ')

export const formatTokens = (value?: number) => {
  if (!value) return '—'
  const binaryMillions = value / 1_048_576
  if (value >= 1_048_576 && Number.isInteger(binaryMillions)) return `${binaryMillions}M`
  if (value >= 1_000_000) {
    const millions = value / 1_000_000
    return `${Number(millions.toFixed(2))}M`
  }
  const binaryThousands = value / 1_024
  if (value % 1_000 !== 0 && Number.isInteger(binaryThousands)) return `${binaryThousands}K`
  if (value >= 1_000) {
    const thousands = value / 1_000
    return `${Number.isInteger(thousands) ? thousands : thousands.toFixed(1)}K`
  }
  return String(value)
}

export const modelContextLabel = (model: CatalogModel) => {
  if (model.limits?.context_window_size) return formatTokens(model.limits.context_window_size)
  if (model.kind === 'virtual') return 'By backend'
  return model.distribution.type === 'proprietary_api' ? 'Undisclosed' : 'Not cataloged'
}

export const modelMaxOutputLabel = (model: CatalogModel) => {
  if (model.limits?.max_output_tokens) return formatTokens(model.limits.max_output_tokens)
  if (model.kind === 'virtual') return 'By backend'
  return model.distribution.type === 'open_weights' ? 'Runtime-set' : 'Provider-defined'
}

export const modelParameterLabel = (model: CatalogModel) => {
  if (model.parameter_size) return model.parameter_size
  if (model.kind === 'virtual') return 'Varies by pool'
  return model.distribution.type === 'proprietary_api' ? 'Undisclosed' : 'Not cataloged'
}

export const formatDate = (value?: string) => {
  if (!value) return 'Not published'
  return new Intl.DateTimeFormat('en', { year: 'numeric', month: 'short' }).format(
    new Date(`${value}T00:00:00Z`),
  )
}

export const formatMetric = (value: number, metric: BenchmarkMetric) => {
  if (metric.unit === 'proportion' || metric.unit === 'fraction') {
    return `${(value * 100).toFixed(1)}%`
  }
  if (metric.unit === 'elo') return Math.round(value).toLocaleString()
  return Number.isInteger(value) ? String(value) : value.toFixed(2)
}

export function Badge({ value, label }: { value: string, label?: string }) {
  return (
    <span className={`${styles.badge} ${styles[`badge_${value}`] ?? ''}`}>
      {label ?? readable(value)}
    </span>
  )
}

export function Tags({ values }: { values: string[] }) {
  return (
    <span className={styles.tags}>
      {values.map(value => (
        <span key={value}>{readable(value)}</span>
      ))}
    </span>
  )
}

export function SelectControl({
  label,
  value,
  options,
  onChange,
}: {
  label: string
  value: string
  options: Array<[string, string]>
  onChange: (value: string) => void
}) {
  return (
    <label className={styles.selectControl}>
      <span>{label}</span>
      <span className={styles.selectShell}>
        <select value={value} onChange={event => onChange(event.target.value)}>
          {options.map(([optionValue, optionLabel]) => (
            <option key={optionValue} value={optionValue}>
              {optionLabel}
            </option>
          ))}
        </select>
        <i aria-hidden="true">⌄</i>
      </span>
    </label>
  )
}

export function Pagination({
  page,
  pageCount,
  total,
  pageSize,
  label,
  onChange,
}: {
  page: number
  pageCount: number
  total: number
  pageSize: number
  label: string
  onChange: (page: number) => void
}) {
  if (total === 0) return null
  const first = (page - 1) * pageSize + 1
  const last = Math.min(page * pageSize, total)
  const pages = Array.from(new Set([1, page - 1, page, page + 1, pageCount])).filter(
    candidate => candidate >= 1 && candidate <= pageCount,
  )
  return (
    <nav className={styles.pagination} aria-label={`${label} pagination`}>
      <span>
        {first}
        –
        {last}
        {' '}
        of
        {total}
      </span>
      <div>
        <button
          type="button"
          disabled={page === 1}
          onClick={() => onChange(page - 1)}
          aria-label="Previous page"
        >
          ←
        </button>
        {pages.map((candidate, index) => (
          <React.Fragment key={candidate}>
            {index > 0 && candidate - pages[index - 1] > 1 ? <span>…</span> : null}
            <button
              type="button"
              className={candidate === page ? styles.currentPage : undefined}
              aria-current={candidate === page ? 'page' : undefined}
              onClick={() => onChange(candidate)}
            >
              {candidate}
            </button>
          </React.Fragment>
        ))}
        <button
          type="button"
          disabled={page === pageCount}
          onClick={() => onChange(page + 1)}
          aria-label="Next page"
        >
          →
        </button>
      </div>
    </nav>
  )
}

export function EmptyState({ title, body }: { title: string, body: string }) {
  return (
    <div className={styles.empty}>
      <strong>{title}</strong>
      <p>{body}</p>
    </div>
  )
}

export const srOnlyClass = styles.srOnly
