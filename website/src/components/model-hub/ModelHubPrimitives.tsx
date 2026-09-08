import React, { useCallback, useEffect, useId, useRef, useState } from 'react'
import { createPortal } from 'react-dom'
import useIsBrowser from '@docusaurus/useIsBrowser'

import type { BenchmarkMetric, CatalogModel } from '../../data/modelHubCatalogTypes'
import { modelHubBenchmarkNormalizedValue } from '../../data/modelHubBenchmarkSupport'
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
  if (metric.normalization || metric.unit === 'proportion' || metric.unit === 'fraction') {
    return `${(modelHubBenchmarkNormalizedValue(value, metric) * 100).toFixed(1)}%`
  }
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
  const isBrowser = useIsBrowser()
  const labelId = useId()
  const [open, setOpen] = useState(false)
  const buttonRef = useRef<HTMLButtonElement>(null)
  const menuRef = useRef<HTMLUListElement>(null)
  const [menuStyle, setMenuStyle] = useState<React.CSSProperties>({})
  const selected = options.find(([optionValue]) => optionValue === value)?.[1] ?? value

  const placeMenu = useCallback(() => {
    const button = buttonRef.current
    if (!button) return
    const rect = button.getBoundingClientRect()
    const gutter = 8
    const width = Math.min(Math.max(rect.width, 14 * 16), window.innerWidth - gutter * 2)
    const spaceBelow = window.innerHeight - rect.bottom - gutter
    const spaceAbove = rect.top - gutter
    const openUp = spaceBelow < 12 * 16 && spaceAbove > spaceBelow
    const maxHeight = Math.max(8 * 16, Math.min(18 * 16, openUp ? spaceAbove : spaceBelow))
    let left = rect.left
    if (left + width > window.innerWidth - gutter) {
      left = Math.max(gutter, window.innerWidth - width - gutter)
    }
    setMenuStyle({
      position: 'fixed',
      zIndex: 10050,
      left,
      width,
      maxHeight,
      ...(openUp
        ? { top: 'auto', bottom: window.innerHeight - rect.top + 4 }
        : { top: rect.bottom + 4, bottom: 'auto' }),
    })
  }, [])

  useEffect(() => {
    if (!open) return
    placeMenu()
    const onPointer = (event: MouseEvent) => {
      const target = event.target as Node
      if (buttonRef.current?.contains(target) || menuRef.current?.contains(target)) return
      setOpen(false)
    }
    const onKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        event.preventDefault()
        setOpen(false)
        buttonRef.current?.focus()
      }
    }
    window.addEventListener('mousedown', onPointer)
    window.addEventListener('keydown', onKey)
    window.addEventListener('resize', placeMenu)
    window.addEventListener('scroll', placeMenu, true)
    return () => {
      window.removeEventListener('mousedown', onPointer)
      window.removeEventListener('keydown', onKey)
      window.removeEventListener('resize', placeMenu)
      window.removeEventListener('scroll', placeMenu, true)
    }
  }, [open, placeMenu])

  return (
    <div className={styles.selectControl}>
      <button
        ref={buttonRef}
        type="button"
        className={styles.selectTrigger}
        aria-haspopup="listbox"
        aria-expanded={open}
        aria-labelledby={`${labelId} ${labelId}-value`}
        onClick={() => setOpen(current => !current)}
      >
        <span id={labelId} className={styles.selectPrefix}>{label}</span>
        <span id={`${labelId}-value`} className={styles.selectValue}>{selected}</span>
        <i aria-hidden="true">⌄</i>
      </button>
      {isBrowser && open
        ? createPortal(
            <ul
              ref={menuRef}
              className={styles.selectMenu}
              role="listbox"
              style={menuStyle}
              aria-labelledby={labelId}
            >
              {options.map(([optionValue, optionLabel]) => (
                <li key={optionValue}>
                  <button
                    type="button"
                    role="option"
                    aria-selected={optionValue === value}
                    className={optionValue === value ? styles.selectOptionActive : undefined}
                    onClick={() => {
                      onChange(optionValue)
                      setOpen(false)
                      buttonRef.current?.focus()
                    }}
                  >
                    {optionLabel}
                  </button>
                </li>
              ))}
            </ul>,
            document.body,
          )
        : null}
    </div>
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
