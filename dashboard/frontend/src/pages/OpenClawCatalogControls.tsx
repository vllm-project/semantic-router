import { useId } from 'react'

import { getOpenClawPageCount, getOpenClawVisibleRange } from '../utils/openClawCatalogSupport'
import styles from './OpenClawPage.module.css'

export interface OpenClawCatalogOption {
  label: string
  value: string
}

interface OpenClawCatalogControlsProps {
  filterLabel: string
  filterOptions: OpenClawCatalogOption[]
  filterValue: string
  itemCount: number
  itemLabel: string
  page: number
  pageSize: number
  searchLabel: string
  searchValue: string
  sortOptions: OpenClawCatalogOption[]
  sortValue: string
  totalCount: number
  onFilterChange: (value: string) => void
  onPageChange: (page: number) => void
  onSearchChange: (value: string) => void
  onSortChange: (value: string) => void
}

export function OpenClawCatalogControls({
  filterLabel,
  filterOptions,
  filterValue,
  itemCount,
  itemLabel,
  page,
  pageSize,
  searchLabel,
  searchValue,
  sortOptions,
  sortValue,
  totalCount,
  onFilterChange,
  onPageChange,
  onSearchChange,
  onSortChange,
}: OpenClawCatalogControlsProps) {
  const searchId = useId()
  const filterId = useId()
  const sortId = useId()

  return (
    <>
      <div className={styles.enterpriseCatalogControls}>
        <label className={styles.enterpriseSearchField} htmlFor={searchId}>
          <span>{searchLabel}</span>
          <input
            id={searchId}
            type="search"
            value={searchValue}
            onChange={(event) => onSearchChange(event.target.value)}
            placeholder={`Search ${itemLabel}`}
          />
        </label>
        <label className={styles.enterpriseSelectField} htmlFor={filterId}>
          <span>{filterLabel}</span>
          <select
            id={filterId}
            value={filterValue}
            onChange={(event) => onFilterChange(event.target.value)}
          >
            {filterOptions.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
        </label>
        <label className={styles.enterpriseSelectField} htmlFor={sortId}>
          <span>Sort</span>
          <select
            id={sortId}
            value={sortValue}
            onChange={(event) => onSortChange(event.target.value)}
          >
            {sortOptions.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
        </label>
      </div>
      <div className={styles.enterpriseCatalogMeta} aria-live="polite">
        <span>
          <strong>{itemCount}</strong> of {totalCount} {itemLabel}
        </span>
        <span>Client view · {pageSize} per page</span>
      </div>
      <OpenClawPagination
        itemCount={itemCount}
        itemLabel={itemLabel}
        page={page}
        pageSize={pageSize}
        onPageChange={onPageChange}
      />
    </>
  )
}

interface OpenClawPaginationProps {
  itemCount: number
  itemLabel: string
  page: number
  pageSize: number
  onPageChange: (page: number) => void
}

export function OpenClawPagination({
  itemCount,
  itemLabel,
  page,
  pageSize,
  onPageChange,
}: OpenClawPaginationProps) {
  const pageCount = getOpenClawPageCount(itemCount, pageSize)
  const safePage = Math.min(Math.max(page, 1), pageCount)
  const range = getOpenClawVisibleRange(itemCount, safePage, pageSize)

  return (
    <nav className={styles.enterprisePagination} aria-label={`${itemLabel} pagination`}>
      <span className={styles.enterprisePageRange}>
        {range.start}–{range.end} of {itemCount} {itemLabel}
      </span>
      <div className={styles.enterprisePageButtons}>
        <button
          type="button"
          onClick={() => onPageChange(1)}
          disabled={safePage === 1}
          aria-label={`First ${itemLabel} page`}
        >
          «
        </button>
        <button
          type="button"
          onClick={() => onPageChange(safePage - 1)}
          disabled={safePage === 1}
          aria-label={`Previous ${itemLabel} page`}
        >
          ‹
        </button>
        <span>
          Page {safePage} of {pageCount}
        </span>
        <button
          type="button"
          onClick={() => onPageChange(safePage + 1)}
          disabled={safePage === pageCount}
          aria-label={`Next ${itemLabel} page`}
        >
          ›
        </button>
        <button
          type="button"
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
