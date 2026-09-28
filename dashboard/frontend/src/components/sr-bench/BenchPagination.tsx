import ProductIcon from '../ProductIcon'
import { number } from './model'
import styles from './SrBench.module.css'

export default function BenchPagination({
  label,
  total,
  page,
  pageSize,
  onChange,
}: {
  label: string
  total: number
  page: number
  pageSize: number
  onChange: (page: number) => void
}) {
  if (total <= pageSize) return null
  const pages = Math.ceil(total / pageSize)
  const current = Math.min(page, pages - 1)
  return (
    <nav className={styles.pagination} aria-label={`${label} pages`}>
      <span role="status">
        {number(current * pageSize + 1)}–{number(Math.min((current + 1) * pageSize, total))} of{' '}
        {number(total)} · Page {current + 1} of {pages}
      </span>
      <div className={styles.actions}>
        <button disabled={current === 0} onClick={() => onChange(current - 1)}>
          <ProductIcon name="chevron-left" /> Previous {label.toLowerCase()}
        </button>
        <button disabled={current + 1 >= pages} onClick={() => onChange(current + 1)}>
          Next {label.toLowerCase()} <ProductIcon name="chevron-right" />
        </button>
      </div>
    </nav>
  )
}
