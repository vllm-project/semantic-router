import type { PropsWithChildren, ReactNode } from 'react'
import type { AccessControlPageState as PageState } from './AccessControlViewTypes'
import { initials, number } from './AccessControlViewSupport'
import styles from './AccessControlPage.module.css'

export function Metric({
  label,
  value,
  detail,
  tone,
}: {
  label: string
  value: string
  detail: string
  tone: string
}) {
  return (
    <article className={`${styles.metric} ${styles[`metric${tone}`]}`}>
      <span>{label}</span>
      <strong>{value}</strong>
      <small>{detail}</small>
    </article>
  )
}

export function PanelHeading({
  eyebrow,
  title,
  aside,
}: {
  eyebrow: string
  title: string
  aside?: string
}) {
  return (
    <div className={styles.panelHeading}>
      <div>
        <span>{eyebrow}</span>
        <h3>{title}</h3>
      </div>
      {aside ? <strong>{aside}</strong> : null}
    </div>
  )
}

export function Avatar({ name, square = false }: { name: string; square?: boolean }) {
  return (
    <span className={`${styles.avatar} ${square ? styles.avatarSquare : ''}`}>
      {initials(name)}
    </span>
  )
}
export function Status({ value, label }: { value: string; label?: string }) {
  return (
    <span
      className={`${styles.status} ${value === 'active' ? styles.statusActive : styles.statusInactive}`}
    >
      <i />
      {label || value}
    </span>
  )
}
export function Quota({ value }: { value: number }) {
  return (
    <span className={value ? styles.quotaValue : styles.quotaUnlimited}>
      {value ? number(value) : 'Unlimited'}
    </span>
  )
}
export function Empty({
  title,
  detail,
  compact = false,
}: {
  title: string
  detail: string
  compact?: boolean
}) {
  return (
    <div className={`${styles.empty} ${compact ? styles.emptyCompact : ''}`}>
      <span>◇</span>
      <strong>{title}</strong>
      <p>{detail}</p>
    </div>
  )
}

export function ListToolbar({
  state,
  onChange,
  placeholder,
  hideSearch = false,
}: {
  state: PageState
  onChange: (value: PageState) => void
  placeholder: string
  hideSearch?: boolean
}) {
  return (
    <div className={styles.listToolbar}>
      {!hideSearch ? (
        <label className={styles.searchBox}>
          <span aria-hidden="true">⌕</span>
          <input
            type="search"
            value={state.query}
            placeholder={placeholder}
            onChange={(event) => onChange({ ...state, query: event.target.value, page: 1 })}
          />
        </label>
      ) : (
        <span />
      )}
      <label className={styles.pageSize}>
        Rows
        <select
          value={state.pageSize}
          onChange={(event) =>
            onChange({ ...state, pageSize: Number(event.target.value), page: 1 })
          }
        >
          <option value="10">10</option>
          <option value="20">20</option>
          <option value="50">50</option>
        </select>
      </label>
    </div>
  )
}

export function Pagination({
  total,
  state,
  onChange,
}: {
  total: number
  state: PageState
  onChange: (value: PageState) => void
}) {
  const pages = Math.max(1, Math.ceil(total / state.pageSize))
  const current = Math.min(state.page, pages)
  const start = total ? (current - 1) * state.pageSize + 1 : 0
  const end = Math.min(total, current * state.pageSize)
  return (
    <div className={styles.pagination}>
      <span>
        {start}–{end} of {number(total)}
      </span>
      <div>
        <button
          type="button"
          disabled={current <= 1}
          onClick={() => onChange({ ...state, page: current - 1 })}
        >
          ←
        </button>
        <span>
          Page {current} of {pages}
        </span>
        <button
          type="button"
          disabled={current >= pages}
          onClick={() => onChange({ ...state, page: current + 1 })}
        >
          →
        </button>
      </div>
    </div>
  )
}

export function EntityTable({
  toolbar,
  pagination,
  children,
}: PropsWithChildren<{ toolbar: ReactNode; pagination: ReactNode }>) {
  return (
    <div className={styles.viewStack}>
      {toolbar}
      <div className={styles.dataTable}>{children}</div>
      {pagination}
    </div>
  )
}
