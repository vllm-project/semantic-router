import { useEffect, useId, useMemo, useRef, useState } from 'react'
import { useInferenceRoutingAccess } from '../contexts/InferenceRoutingAccessContext'
import type { SelfInferenceKey } from '../utils/routerManagementTypes'
import {
  fetchSelfInferenceKeyPage,
  mergeSelfInferenceKeyPages,
  SELF_KEY_RENDER_LIMIT,
  SELF_KEY_SEARCH_DEBOUNCE_MS,
} from '../utils/selfInferenceKeys'
import ProductIcon from './ProductIcon'
import styles from './InferenceKeySelector.module.css'

interface InferenceKeySelectorProps {
  className?: string
  disabled?: boolean
  label?: string
}

function keyLabel(key: SelfInferenceKey): string {
  return key.name || key.keyId
}

export default function InferenceKeySelector({
  className = '',
  disabled = false,
  label = 'API key',
}: InferenceKeySelectorProps) {
  const { keys, keysHasMore, keysStatus, selectedKey, selectedKeyId, selectKey } =
    useInferenceRoutingAccess()
  const instanceID = useId()
  const listboxID = `${instanceID}-options`
  const labelID = `${instanceID}-label`
  const containerRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLInputElement>(null)
  const paginationControllerRef = useRef<AbortController | null>(null)
  const [open, setOpen] = useState(false)
  const [search, setSearch] = useState('')
  const [options, setOptions] = useState<SelfInferenceKey[]>([])
  const [nextCursor, setNextCursor] = useState('')
  const [hasMore, setHasMore] = useState(false)
  const [loading, setLoading] = useState(false)
  const [loadingMore, setLoadingMore] = useState(false)
  const [error, setError] = useState('')
  const [activeIndex, setActiveIndex] = useState(-1)

  const canChooseAnother = keys.length > 1 || keysHasMore
  const displayedValue = open ? search : selectedKey ? keyLabel(selectedKey) : ''
  const activeOptionID =
    activeIndex >= 0 && options[activeIndex]
      ? `${instanceID}-option-${options[activeIndex].keyId}`
      : undefined
  const reachedRenderLimit = options.length >= SELF_KEY_RENDER_LIMIT && hasMore

  const close = (restoreFocus = false) => {
    paginationControllerRef.current?.abort()
    setOpen(false)
    setSearch('')
    setLoading(false)
    setLoadingMore(false)
    setError('')
    setActiveIndex(-1)
    if (restoreFocus) requestAnimationFrame(() => inputRef.current?.focus())
  }

  const openSelector = () => {
    if (disabled || keysStatus !== 'ready') return
    if (open) return
    setOpen(true)
    setSearch('')
    setOptions(keys)
    setHasMore(keysHasMore)
    setNextCursor('')
    setActiveIndex(
      Math.max(
        0,
        keys.findIndex((key) => key.keyId === selectedKeyId),
      ),
    )
  }

  useEffect(() => {
    if (!open) return
    const controller = new AbortController()
    setLoading(true)
    setError('')
    const timer = window.setTimeout(
      () => {
        void fetchSelfInferenceKeyPage({ search }, controller.signal)
          .then((page) => {
            if (controller.signal.aborted) return
            setOptions(page.items)
            setNextCursor(page.nextCursor ?? '')
            setHasMore(page.hasMore)
            setActiveIndex(page.items.length ? 0 : -1)
          })
          .catch((cause: unknown) => {
            if (controller.signal.aborted) return
            setOptions([])
            setNextCursor('')
            setHasMore(false)
            setActiveIndex(-1)
            setError(cause instanceof Error ? cause.message : 'Could not load API keys.')
          })
          .finally(() => {
            if (!controller.signal.aborted) setLoading(false)
          })
      },
      search ? SELF_KEY_SEARCH_DEBOUNCE_MS : 0,
    )
    return () => {
      window.clearTimeout(timer)
      controller.abort()
    }
  }, [open, search])

  useEffect(() => {
    if (!open) return
    const onPointerDown = (event: PointerEvent) => {
      if (!containerRef.current?.contains(event.target as Node)) close()
    }
    document.addEventListener('pointerdown', onPointerDown)
    return () => document.removeEventListener('pointerdown', onPointerDown)
  }, [open])

  useEffect(
    () => () => {
      paginationControllerRef.current?.abort()
    },
    [],
  )

  const choose = (key: SelfInferenceKey) => {
    selectKey(key)
    close(true)
  }

  const loadMore = async () => {
    if (!nextCursor || loadingMore || reachedRenderLimit) return
    paginationControllerRef.current?.abort()
    const controller = new AbortController()
    paginationControllerRef.current = controller
    setLoadingMore(true)
    setError('')
    try {
      const page = await fetchSelfInferenceKeyPage(
        { search, cursor: nextCursor },
        controller.signal,
      )
      if (controller.signal.aborted) return
      setOptions((current) => mergeSelfInferenceKeyPages(current, page.items))
      setNextCursor(page.nextCursor ?? '')
      setHasMore(page.hasMore)
    } catch (cause) {
      if (!controller.signal.aborted) {
        setError(cause instanceof Error ? cause.message : 'Could not load more API keys.')
      }
    } finally {
      if (!controller.signal.aborted) setLoadingMore(false)
    }
  }

  const keyDown = (event: React.KeyboardEvent<HTMLInputElement>) => {
    if (event.key === 'Escape') {
      if (open) {
        event.preventDefault()
        close(true)
      }
      return
    }
    if (!open && (event.key === 'ArrowDown' || event.key === 'Enter' || event.key === ' ')) {
      event.preventDefault()
      openSelector()
      return
    }
    if (!open || !options.length) return
    if (event.key === 'ArrowDown') {
      event.preventDefault()
      setActiveIndex((current) => (current + 1 + options.length) % options.length)
    } else if (event.key === 'ArrowUp') {
      event.preventDefault()
      setActiveIndex((current) => (current - 1 + options.length) % options.length)
    } else if (event.key === 'Enter' && activeIndex >= 0) {
      event.preventDefault()
      choose(options[activeIndex])
    }
  }

  const status = useMemo(() => {
    if (loading) return 'Searching…'
    if (error) return error
    if (!options.length) return 'No matching API keys'
    return ''
  }, [error, loading, options.length])

  if (!canChooseAnother) return null

  return (
    <div ref={containerRef} className={`${styles.control} ${className}`}>
      <span id={labelID} className={styles.label}>
        <ProductIcon name="key" aria-hidden="true" />
        {label}
      </span>
      <div className={styles.combobox}>
        <input
          ref={inputRef}
          role="combobox"
          aria-labelledby={labelID}
          aria-expanded={open}
          aria-controls={listboxID}
          aria-activedescendant={activeOptionID}
          aria-autocomplete="list"
          autoComplete="off"
          value={displayedValue}
          placeholder={open ? 'Search API keys' : undefined}
          readOnly={!open}
          disabled={disabled || keysStatus !== 'ready'}
          onClick={openSelector}
          onChange={(event) => {
            setSearch(event.target.value)
            setOptions([])
            setNextCursor('')
            setHasMore(false)
            setActiveIndex(-1)
          }}
          onKeyDown={keyDown}
        />
        <button
          type="button"
          className={styles.toggle}
          aria-label={open ? `Close ${label} selector` : `Open ${label} selector`}
          disabled={disabled || keysStatus !== 'ready'}
          onClick={() => {
            if (open) close(true)
            else {
              openSelector()
              requestAnimationFrame(() => inputRef.current?.focus())
            }
          }}
        >
          <ProductIcon name="chevron-down" aria-hidden="true" />
        </button>
      </div>
      {open ? (
        <div className={styles.popover}>
          <div id={listboxID} className={styles.listbox} role="listbox" aria-label={label}>
            {options.map((key, index) => (
              <button
                id={`${instanceID}-option-${key.keyId}`}
                key={key.keyId}
                type="button"
                role="option"
                aria-selected={key.keyId === selectedKeyId}
                className={index === activeIndex ? styles.activeOption : undefined}
                onMouseMove={() => setActiveIndex(index)}
                onClick={() => choose(key)}
              >
                <span>{keyLabel(key)}</span>
                <small>{key.owner.type === 'team' ? 'Team key' : 'Personal key'}</small>
                {key.keyId === selectedKeyId ? (
                  <ProductIcon name="check" aria-hidden="true" />
                ) : null}
              </button>
            ))}
          </div>
          {status ? (
            <p className={error ? styles.error : styles.status} role={error ? 'alert' : 'status'}>
              {status}
            </p>
          ) : null}
          {hasMore && !reachedRenderLimit ? (
            <button
              type="button"
              className={styles.loadMore}
              disabled={loadingMore || !nextCursor}
              onClick={() => void loadMore()}
            >
              {loadingMore ? 'Loading…' : 'Load more'}
            </button>
          ) : null}
          {reachedRenderLimit ? <p className={styles.status}>Type to narrow results.</p> : null}
        </div>
      ) : null}
    </div>
  )
}
