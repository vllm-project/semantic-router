import type { PropsWithChildren, ReactNode } from 'react'
import ProductIcon from '../components/ProductIcon'
import type { AccessControlPageState as PageState } from './AccessControlViewTypes'
import { initials, number } from './AccessControlViewSupport'
import styles from './AccessControlPage.module.css'

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
      <span>
        <ProductIcon name="inbox" />
      </span>
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
          <ProductIcon name="search" />
          <input
            type="search"
            value={state.query}
            placeholder={placeholder}
            maxLength={200}
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
          <ProductIcon name="chevron-left" />
        </button>
        <span>
          Page {current} of {pages}
        </span>
        <button
          type="button"
          disabled={current >= pages}
          onClick={() => onChange({ ...state, page: current + 1 })}
        >
          <ProductIcon name="chevron-right" />
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
