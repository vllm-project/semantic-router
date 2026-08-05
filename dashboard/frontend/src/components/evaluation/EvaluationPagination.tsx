import styles from './EvaluationPagination.module.css'

interface EvaluationPaginationProps {
  page: number
  totalPages: number
  start: number
  end: number
  total: number
  itemLabel: string
  onPageChange: (page: number) => void
}

export function EvaluationPagination({
  page,
  totalPages,
  start,
  end,
  total,
  itemLabel,
  onPageChange,
}: EvaluationPaginationProps) {
  if (total === 0) return null

  return (
    <nav className={styles.pagination} aria-label={`${itemLabel} pagination`}>
      <span className={styles.summary}>
        {start + 1}–{end} of {total} {itemLabel}
      </span>
      <div className={styles.controls}>
        <button type="button" onClick={() => onPageChange(1)} disabled={page === 1} aria-label="First page">
          «
        </button>
        <button type="button" onClick={() => onPageChange(Math.max(1, page - 1))} disabled={page === 1}>
          Previous
        </button>
        <span>Page {page} of {totalPages}</span>
        <button
          type="button"
          onClick={() => onPageChange(Math.min(totalPages, page + 1))}
          disabled={page === totalPages}
        >
          Next
        </button>
        <button
          type="button"
          onClick={() => onPageChange(totalPages)}
          disabled={page === totalPages}
          aria-label="Last page"
        >
          »
        </button>
      </div>
    </nav>
  )
}

export default EvaluationPagination
