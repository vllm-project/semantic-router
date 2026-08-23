import { useCallback, useEffect, useMemo, useRef, useState } from 'react'

import { routingManagementApi } from '../utils/routingManagementApi'
import ProductIcon from './ProductIcon'
import styles from './AgentManagementPanel.module.css'

interface AgentCapabilityPickerProps {
  label: string
  selected: string[]
  onChange: (capabilities: string[]) => void
  onAvailabilityChange?: (available: boolean) => void
}

function describeLoadError(cause: unknown): string {
  if (cause instanceof DOMException && cause.name === 'AbortError') return ''
  return cause instanceof Error ? cause.message : 'Capabilities are unavailable.'
}

export default function AgentCapabilityPicker({
  label,
  selected,
  onChange,
  onAvailabilityChange,
}: AgentCapabilityPickerProps) {
  const [available, setAvailable] = useState<string[]>([])
  const [query, setQuery] = useState('')
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [requestRevision, setRequestRevision] = useState(0)
  const requestGeneration = useRef(0)

  const load = useCallback(async (signal: AbortSignal) => {
    const generation = ++requestGeneration.current
    setLoading(true)
    setError(null)
    try {
      const cards = await routingManagementApi.listModelCards(signal)
      if (signal.aborted || generation !== requestGeneration.current) return
      setAvailable(
        [...new Set(cards.flatMap((model) => model.card.capabilities))]
          .filter(Boolean)
          .sort((left, right) => left.localeCompare(right)),
      )
    } catch (cause) {
      if (signal.aborted || generation !== requestGeneration.current) return
      const message = describeLoadError(cause)
      if (message) setError(message)
    } finally {
      if (!signal.aborted && generation === requestGeneration.current) setLoading(false)
    }
  }, [])

  useEffect(() => {
    const controller = new AbortController()
    void load(controller.signal)
    return () => {
      controller.abort()
      requestGeneration.current += 1
    }
  }, [load, requestRevision])

  useEffect(() => {
    onAvailabilityChange?.(!loading && !error)
  }, [error, loading, onAvailabilityChange])

  const options = useMemo(() => {
    const normalizedQuery = query.trim().toLocaleLowerCase()
    return [...new Set([...selected, ...available])]
      .filter(
        (capability) =>
          selected.includes(capability) || capability.toLocaleLowerCase().includes(normalizedQuery),
      )
      .sort((left, right) => {
        const leftSelected = selected.includes(left)
        const rightSelected = selected.includes(right)
        if (leftSelected !== rightSelected) return leftSelected ? -1 : 1
        return left.localeCompare(right)
      })
  }, [available, query, selected])

  const toggle = (capability: string) => {
    onChange(
      selected.includes(capability)
        ? selected.filter((item) => item !== capability)
        : [...selected, capability],
    )
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
      {error ? (
        <p className={styles.pickerError} role="alert">
          <ProductIcon name="alert" />
          {error}
          <button type="button" onClick={() => setRequestRevision((current) => current + 1)}>
            Retry
          </button>
        </p>
      ) : null}
      <div className={styles.pickerList}>
        {loading && options.length === 0 ? (
          <span className={styles.emptyInline} role="status">
            Loading…
          </span>
        ) : null}
        {options.map((capability) => {
          const checked = selected.includes(capability)
          const unavailable = !available.includes(capability)
          return (
            <label
              key={capability}
              className={`${styles.pickerItem} ${checked ? styles.pickerItemSelected : ''}`}
            >
              <input type="checkbox" checked={checked} onChange={() => toggle(capability)} />
              <span>
                <strong>{capability}</strong>
                {unavailable && !loading ? <small>Not currently advertised</small> : null}
              </span>
              <ProductIcon name={checked ? 'check' : 'plus'} />
            </label>
          )
        })}
        {!loading && !error && options.length === 0 ? (
          <span className={styles.emptyInline}>No capabilities found.</span>
        ) : null}
      </div>
    </div>
  )
}
