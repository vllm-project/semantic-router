import { type ReactNode, useCallback, useEffect, useMemo, useRef, useState } from 'react'
import ProductIcon from '../components/ProductIcon'
import type { AccessListParams, AccessPage } from '../utils/inferenceAccessApi'
import {
  ACCESS_PICKER_HYDRATION_CONCURRENCY,
  accessPickerRequest,
  mergeAccessPickerPage,
  missingSelectedPickerIds,
} from './accessAsyncResourcePickerSupport'
import styles from './AccessControlPage.module.css'

export interface AccessPickerSource<T> {
  list: (params: AccessListParams) => Promise<AccessPage<T>>
  detail: (id: string) => Promise<T>
  id: (item: T) => string
  title: (item: T) => string
  description: (item: T) => string
}

interface Props<T> {
  ariaLabel: string
  selectedIds: string[]
  source: AccessPickerSource<T>
  onChange: (ids: string[]) => void
  multiple?: boolean
  optional?: boolean
  placeholder?: string
  emptyText?: string
  optionalTitle?: string
  optionalDescription?: string
  renderSelectedDetail?: (item: T) => ReactNode
  compact?: boolean
  compactEmptyLabel?: string
}

const SEARCH_DELAY_MS = 240

export default function AccessAsyncResourcePicker<T>({
  ariaLabel,
  selectedIds,
  source,
  onChange,
  multiple = false,
  optional = false,
  placeholder = 'Search',
  emptyText = 'No matches',
  optionalTitle = 'Inherit',
  optionalDescription = 'Use the owner’s effective policy',
  renderSelectedDetail,
  compact = false,
  compactEmptyLabel = 'All',
}: Props<T>) {
  const [search, setSearch] = useState('')
  const [debouncedSearch, setDebouncedSearch] = useState('')
  const [items, setItems] = useState<T[]>([])
  const [hydrated, setHydrated] = useState<Record<string, T>>({})
  const [failedHydration, setFailedHydration] = useState<Record<string, true>>({})
  const [nextCursor, setNextCursor] = useState<string>()
  const [hasMore, setHasMore] = useState(false)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [expanded, setExpanded] = useState(!compact)
  const generation = useRef(0)
  const searchInput = useRef<HTMLInputElement>(null)

  useEffect(() => {
    const timeout = window.setTimeout(() => setDebouncedSearch(search.trim()), SEARCH_DELAY_MS)
    return () => window.clearTimeout(timeout)
  }, [search])

  useEffect(() => {
    if (compact && expanded) searchInput.current?.focus()
  }, [compact, expanded])

  const load = useCallback(
    async (cursor?: string) => {
      const requestGeneration = cursor ? generation.current : generation.current + 1
      if (!cursor) generation.current = requestGeneration
      setLoading(true)
      setError('')
      try {
        const page = await source.list(accessPickerRequest(debouncedSearch, cursor))
        if (requestGeneration !== generation.current) return
        setItems((current) => {
          return mergeAccessPickerPage(current, page.items, Boolean(cursor), source.id)
        })
        setNextCursor(page.nextCursor)
        setHasMore(page.hasMore)
      } catch (nextError) {
        if (requestGeneration !== generation.current) return
        setError(nextError instanceof Error ? nextError.message : 'Could not load options')
      } finally {
        if (requestGeneration === generation.current) setLoading(false)
      }
    },
    [debouncedSearch, source],
  )

  useEffect(() => {
    if (compact && !expanded) return
    void load()
  }, [compact, expanded, load])

  useEffect(() => {
    let cancelled = false
    const missing = missingSelectedPickerIds(
      selectedIds,
      items,
      hydrated,
      failedHydration,
      source.id,
    )
    if (!missing.length) return

    void (async () => {
      for (let offset = 0; offset < missing.length; offset += ACCESS_PICKER_HYDRATION_CONCURRENCY) {
        const batch = missing.slice(offset, offset + ACCESS_PICKER_HYDRATION_CONCURRENCY)
        const results = await Promise.allSettled(batch.map((id) => source.detail(id)))
        if (cancelled) return
        setHydrated((current) => {
          const next = { ...current }
          results.forEach((result, index) => {
            if (result.status === 'fulfilled') next[batch[index]] = result.value
          })
          return next
        })
        setFailedHydration((current) => {
          const next = { ...current }
          results.forEach((result, index) => {
            if (result.status === 'rejected') next[batch[index]] = true
          })
          return next
        })
      }
    })()
    return () => {
      cancelled = true
    }
  }, [failedHydration, hydrated, items, selectedIds, source])

  const selectedItems = useMemo(
    () =>
      selectedIds
        .map((id) => items.find((item) => source.id(item) === id) || hydrated[id])
        .filter((item): item is T => Boolean(item)),
    [hydrated, items, selectedIds, source],
  )
  const visibleItems = useMemo(() => {
    const selected = new Set(selectedIds)
    return [...selectedItems, ...items.filter((item) => !selected.has(source.id(item)))]
  }, [items, selectedIds, selectedItems, source])
  const compactLabel = selectedItems[0]
    ? `${source.title(selectedItems[0])}${selectedIds.length > 1 ? ` +${selectedIds.length - 1}` : ''}`
    : selectedIds.length
      ? `${selectedIds.length} selected`
      : compactEmptyLabel

  const toggle = (id: string) => {
    if (!multiple) {
      onChange(selectedIds[0] === id && optional ? [] : [id])
      if (compact) {
        setSearch('')
        setExpanded(false)
      }
      return
    }
    onChange(
      selectedIds.includes(id) ? selectedIds.filter((value) => value !== id) : [...selectedIds, id],
    )
  }

  return (
    <div
      className={`${styles.asyncPicker} ${compact ? styles.asyncPickerCompact : ''}`}
      onBlur={(event) => {
        if (compact && !event.currentTarget.contains(event.relatedTarget as Node | null)) {
          setExpanded(false)
        }
      }}
      onKeyDown={(event) => {
        if (compact && event.key === 'Escape') {
          event.preventDefault()
          setExpanded(false)
        }
      }}
    >
      {compact && !expanded ? (
        <button
          type="button"
          className={styles.asyncPickerCompactValue}
          onClick={() => setExpanded(true)}
          aria-haspopup="listbox"
          aria-expanded="false"
        >
          <span>{compactLabel}</span>
          <ProductIcon name="chevron-down" aria-hidden="true" />
        </button>
      ) : (
        <label className={styles.asyncPickerSearch}>
          <ProductIcon name="search" aria-hidden="true" />
          <input
            ref={searchInput}
            type="search"
            value={search}
            onChange={(event) => setSearch(event.target.value)}
            placeholder={placeholder}
            aria-label={ariaLabel}
            autoComplete="off"
            maxLength={200}
          />
          {search ? (
            <button type="button" onClick={() => setSearch('')} aria-label="Clear search">
              <ProductIcon name="close" />
            </button>
          ) : null}
        </label>
      )}
      {!compact || expanded ? (
        <div className={styles.asyncPickerMenu}>
          <div
            className={styles.asyncPickerList}
            role="listbox"
            aria-multiselectable={multiple || undefined}
            aria-busy={loading}
          >
            {optional && !multiple ? (
              <button
                type="button"
                role="option"
                aria-selected={!selectedIds.length}
                className={!selectedIds.length ? styles.asyncPickerOptionSelected : ''}
                onClick={() => {
                  onChange([])
                  if (compact) setExpanded(false)
                }}
              >
                <span>
                  <strong>{optionalTitle}</strong>
                  <small>{optionalDescription}</small>
                </span>
                {!selectedIds.length ? <ProductIcon name="check" aria-hidden="true" /> : null}
              </button>
            ) : null}
            {visibleItems.map((item) => {
              const id = source.id(item)
              const selected = selectedIds.includes(id)
              return (
                <button
                  type="button"
                  role="option"
                  aria-selected={selected}
                  className={selected ? styles.asyncPickerOptionSelected : ''}
                  key={id}
                  onClick={() => toggle(id)}
                >
                  <span>
                    <strong>{source.title(item)}</strong>
                    <small>{source.description(item)}</small>
                  </span>
                  {selected ? <ProductIcon name="check" aria-hidden="true" /> : null}
                </button>
              )
            })}
            {!visibleItems.length && !loading && !error ? <p>{emptyText}</p> : null}
          </div>
          {renderSelectedDetail && selectedItems.length ? (
            <div className={styles.asyncPickerSelectedDetails}>
              {selectedItems.map((item) => (
                <div key={source.id(item)}>{renderSelectedDetail(item)}</div>
              ))}
            </div>
          ) : null}
          {selectedIds.some(
            (id) => failedHydration[id] && !items.some((item) => source.id(item) === id),
          ) ? (
            <p className={styles.asyncPickerHydrationError} role="status">
              A selected item is unavailable. Its ID remains saved.
            </p>
          ) : null}
          <div className={styles.asyncPickerFooter} aria-live="polite">
            <span>
              {error
                ? 'Options unavailable'
                : loading
                  ? 'Loading…'
                  : `${selectedIds.length} selected`}
            </span>
            {error ? (
              <button type="button" onClick={() => void load()} disabled={loading}>
                <ProductIcon name="refresh" /> Retry
              </button>
            ) : hasMore && nextCursor ? (
              <button type="button" onClick={() => void load(nextCursor)} disabled={loading}>
                <ProductIcon name="chevron-down" /> More
              </button>
            ) : null}
          </div>
        </div>
      ) : null}
    </div>
  )
}
