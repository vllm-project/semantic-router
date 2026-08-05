import styles from './MCPConfigPanel.module.css'
import { getMCPPageCount, getMCPVisibleRange } from './mcpConfigPanelUtils'

interface MCPListPaginationProps {
  itemCount: number
  itemLabel: string
  page: number
  pageSize: number
  onPageChange: (page: number) => void
}

export function MCPListPagination({
  itemCount,
  itemLabel,
  page,
  pageSize,
  onPageChange,
}: MCPListPaginationProps) {
  const pageCount = getMCPPageCount(itemCount, pageSize)
  const safePage = Math.min(Math.max(1, page), pageCount)
  const range = getMCPVisibleRange(itemCount, safePage, pageSize)

  return (
    <nav className={styles.pagination} aria-label={`${itemLabel} pagination`}>
      <span className={styles.pageRange} aria-live="polite">
        {range.start}–{range.end} of {itemCount} {itemLabel}
      </span>
      <div className={styles.pageControls}>
        <button
          type="button"
          className={styles.pageBtn}
          onClick={() => onPageChange(1)}
          disabled={safePage === 1}
          aria-label={`First ${itemLabel} page`}
        >
          «
        </button>
        <button
          type="button"
          className={styles.pageBtn}
          onClick={() => onPageChange(safePage - 1)}
          disabled={safePage === 1}
          aria-label={`Previous ${itemLabel} page`}
        >
          ‹
        </button>
        <span className={styles.pageInfo}>
          Page {safePage} of {pageCount}
        </span>
        <button
          type="button"
          className={styles.pageBtn}
          onClick={() => onPageChange(safePage + 1)}
          disabled={safePage === pageCount}
          aria-label={`Next ${itemLabel} page`}
        >
          ›
        </button>
        <button
          type="button"
          className={styles.pageBtn}
          onClick={() => onPageChange(pageCount)}
          disabled={safePage === pageCount}
          aria-label={`Last ${itemLabel} page`}
        >
          »
        </button>
      </div>
    </nav>
  )
}
