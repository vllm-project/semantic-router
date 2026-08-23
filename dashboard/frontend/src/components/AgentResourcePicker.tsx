import { useEffect, useMemo, useRef, useState } from 'react'

import type { AgentPage } from '../generated/managementApiContract'
import ProductIcon from './ProductIcon'
import styles from './AgentManagementPanel.module.css'

interface AgentResourcePickerProps<T> {
  label: string
  selectedIds: string[]
  loadPage: (search?: string, cursor?: string, signal?: AbortSignal) => Promise<AgentPage<T>>
  resolveSelected?: (id: string, signal?: AbortSignal) => Promise<T | null>
  getId: (item: T) => string
  getLabel: (item: T) => string
  getDescription?: (item: T) => string | undefined
  onChange: (items: T[]) => void
  onAvailabilityChange?: (available: boolean) => void
}

export default function AgentResourcePicker<T>({
  label,
  selectedIds,
  loadPage,
  resolveSelected,
  getId,
  getLabel,
  getDescription,
  onChange,
  onAvailabilityChange,
}: AgentResourcePickerProps<T>) {
  const [items, setItems] = useState<Map<string, T>>(new Map())
  const [query, setQuery] = useState('')
  const [cursor, setCursor] = useState<string | undefined>()
  const [hasMore, setHasMore] = useState(false)
  const [loading, setLoading] = useState(true)
  const [pageReady, setPageReady] = useState(false)
  const [hydrating, setHydrating] = useState(false)
  const [pageError, setPageError] = useState<string | null>(null)
  const [hydrationError, setHydrationError] = useState<string | null>(null)
  const generation = useRef(0)
  const pageController = useRef<AbortController | null>(null)
  const loadPageRef = useRef(loadPage)
  const resolveSelectedRef = useRef(resolveSelected)
  const getIdRef = useRef(getId)
  const availabilityRef = useRef(onAvailabilityChange)
  const selectedIdsRef = useRef(selectedIds)
  const selectedKey = selectedIds.join('\u0000')

  useEffect(() => {
    loadPageRef.current = loadPage
  }, [loadPage])
  useEffect(() => {
    resolveSelectedRef.current = resolveSelected
  }, [resolveSelected])
  useEffect(() => {
    getIdRef.current = getId
  }, [getId])
  useEffect(() => {
    availabilityRef.current = onAvailabilityChange
  }, [onAvailabilityChange])
  useEffect(() => {
    selectedIdsRef.current = selectedIds
  }, [selectedIds])

  useEffect(() => {
    availabilityRef.current?.(pageReady && !hydrating && !pageError && !hydrationError)
  }, [hydrating, hydrationError, pageError, pageReady])

  const requestPage = async (nextCursor?: string) => {
    const requestGeneration = ++generation.current
    pageController.current?.abort()
    const controller = new AbortController()
    pageController.current = controller
    setLoading(true)
    if (!nextCursor) setPageReady(false)
    try {
      const page = await loadPageRef.current(
        query.trim() || undefined,
        nextCursor,
        controller.signal,
      )
      if (controller.signal.aborted || generation.current !== requestGeneration) return
      setItems((current) => {
        const next = nextCursor ? new Map(current) : new Map<string, T>()
        page.data.forEach((item) => next.set(getIdRef.current(item), item))
        selectedIdsRef.current.forEach((id) => {
          const selected = current.get(id)
          if (selected) next.set(id, selected)
        })
        return next
      })
      setCursor(page.page.nextCursor)
      setHasMore(page.page.hasMore)
      setPageError(null)
      setPageReady(true)
    } catch (cause) {
      if (controller.signal.aborted || generation.current !== requestGeneration) return
      setPageError(cause instanceof Error ? cause.message : `${label} are unavailable.`)
    } finally {
      if (!controller.signal.aborted && generation.current === requestGeneration) setLoading(false)
      if (pageController.current === controller) pageController.current = null
    }
  }

  useEffect(() => {
    const timer = window.setTimeout(() => void requestPage(), 180)
    return () => {
      window.clearTimeout(timer)
      pageController.current?.abort()
      generation.current += 1
    }
  }, [query]) // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    const resolve = resolveSelectedRef.current
    setHydrationError(null)
    setHydrating(false)
    const missing = selectedIds.filter((id) => !items.has(id))
    if (!missing.length) return
    if (!resolve) {
      setHydrationError(`Selected ${label.toLowerCase()} could not be loaded.`)
      return
    }
    let active = true
    const controller = new AbortController()
    setHydrating(true)
    void Promise.all(missing.map((id) => resolve(id, controller.signal)))
      .then((resolved) => {
        if (!active) return
        const unresolved = resolved.filter((item) => item === null).length
        setItems((current) => {
          const next = new Map(current)
          resolved.forEach((item) => {
            if (item) next.set(getIdRef.current(item), item)
          })
          return next
        })
        setHydrationError(
          unresolved > 0 ? `A selected ${label.toLowerCase()} item is no longer available.` : null,
        )
      })
      .catch((cause: unknown) => {
        if (!active) return
        setHydrationError(
          cause instanceof Error
            ? cause.message
            : `Selected ${label.toLowerCase()} could not be loaded.`,
        )
      })
      .finally(() => {
        if (active) setHydrating(false)
      })
    return () => {
      active = false
      controller.abort()
    }
  }, [selectedKey]) // eslint-disable-line react-hooks/exhaustive-deps

  const ordered = useMemo(
    () =>
      [...items.values()].sort((left, right) => {
        const leftSelected = selectedIds.includes(getId(left))
        const rightSelected = selectedIds.includes(getId(right))
        if (leftSelected !== rightSelected) return leftSelected ? -1 : 1
        return getLabel(left).localeCompare(getLabel(right))
      }),
    [getId, getLabel, items, selectedIds],
  )

  const toggle = (item: T) => {
    const id = getId(item)
    const nextIds = selectedIds.includes(id)
      ? selectedIds.filter((value) => value !== id)
      : [...selectedIds, id]
    onChange(nextIds.map((value) => items.get(value)).filter((value): value is T => Boolean(value)))
  }

  return (
    <div className={styles.resourcePicker} aria-busy={loading}>
      <label className={styles.pickerSearch}>
        <ProductIcon name="search" />
        <span className={styles.srOnly}>Search {label}</span>
        <input
          type="search"
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          placeholder={`Search ${label.toLowerCase()}`}
        />
      </label>
      {pageError ? (
        <p className={styles.pickerError} role="alert">
          <ProductIcon name="alert" />
          {pageError}
          <button type="button" onClick={() => void requestPage()}>
            Retry
          </button>
        </p>
      ) : null}
      {hydrationError ? (
        <p className={styles.pickerError} role="alert">
          <ProductIcon name="alert" />
          {hydrationError}
        </p>
      ) : null}
      <div className={styles.pickerList}>
        {loading && ordered.length === 0 ? (
          <span className={styles.emptyInline} role="status">
            Loading…
          </span>
        ) : null}
        {ordered.map((item) => {
          const id = getId(item)
          const checked = selectedIds.includes(id)
          return (
            <label
              key={id}
              className={`${styles.pickerItem} ${checked ? styles.pickerItemSelected : ''}`}
            >
              <input type="checkbox" checked={checked} onChange={() => toggle(item)} />
              <span>
                <strong>{getLabel(item)}</strong>
                {getDescription ? <small>{getDescription(item)}</small> : null}
              </span>
              <ProductIcon name={checked ? 'check' : 'plus'} />
            </label>
          )
        })}
        {!loading && !pageError && ordered.length === 0 ? (
          <span className={styles.emptyInline}>No matches.</span>
        ) : null}
      </div>
      {hasMore ? (
        <button
          type="button"
          className={styles.pickerMore}
          onClick={() => void requestPage(cursor)}
          disabled={loading}
        >
          {loading ? 'Loading…' : 'Load more'}
        </button>
      ) : null}
    </div>
  )
}
