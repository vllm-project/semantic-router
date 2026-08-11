import React, { useCallback, useEffect, useMemo, useState } from 'react'
import { useReadonly } from '../contexts/ReadonlyContext'
import styles from './ResponseCachePage.module.css'

interface CacheCapabilities {
  backend: string
  exact: boolean
  semantic: boolean
  per_entry_ttl: boolean
  invalidate: boolean
  flush: boolean
  shared: boolean
  consistency_model: string
}

interface CacheStats {
  total_entries: number
  exact_hit_count: number
  semantic_hit_count: number
  l1_hit_count: number
  l2_hit_count: number
  miss_count: number
  l1_entries: number
  singleflight_waiters: number
  invalidation_count: number
  stale_miss_count: number
  fail_open_count: number
}

const endpoint = (path: string) => `/api/router/api/v1/response-cache/${path}`

const ResponseCachePage: React.FC = () => {
  const { isReadonly } = useReadonly()
  const [capabilities, setCapabilities] = useState<CacheCapabilities | null>(null)
  const [stats, setStats] = useState<CacheStats | null>(null)
  const [health, setHealth] = useState('loading')
  const [auditEntries, setAuditEntries] = useState<Array<Record<string, unknown>>>([])
  const [error, setError] = useState<string | null>(null)
  const [recipe, setRecipe] = useState('')
  const [model, setModel] = useState('')
  const [protocol, setProtocol] = useState('')
  const [dryRun, setDryRun] = useState(true)
  const [confirm, setConfirm] = useState('')
  const [actionResult, setActionResult] = useState<string | null>(null)

  const refresh = useCallback(async () => {
    try {
      const [capabilitiesResponse, healthResponse, statsResponse, auditResponse] =
        await Promise.all([
          fetch(endpoint('capabilities')),
          fetch(endpoint('health')),
          fetch(endpoint('stats')),
          fetch(endpoint('audit')),
        ])
      if (!capabilitiesResponse.ok || !statsResponse.ok) {
        throw new Error('Response cache management API is unavailable')
      }
      setCapabilities((await capabilitiesResponse.json()) as CacheCapabilities)
      setStats((await statsResponse.json()) as CacheStats)
      if (auditResponse.ok) {
        const audit = (await auditResponse.json()) as { entries?: Array<Record<string, unknown>> }
        setAuditEntries(audit.entries ?? [])
      }
      setHealth(healthResponse.ok ? 'healthy' : 'unhealthy')
      setError(null)
    } catch (requestError) {
      setHealth('unavailable')
      setError(requestError instanceof Error ? requestError.message : 'Unable to load cache status')
    }
  }, [])

  useEffect(() => {
    void refresh()
  }, [refresh])

  const hitRate = useMemo(() => {
    if (!stats) return 0
    const hits = stats.exact_hit_count + stats.semantic_hit_count
    const total = hits + stats.miss_count
    return total === 0 ? 0 : (hits / total) * 100
  }, [stats])

  const invalidate = async () => {
    const partition =
      recipe || model || protocol ? { recipe, request_model: model, protocol } : undefined
    const response = await fetch(endpoint('invalidate'), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        selector: { partition, epoch: partition ? undefined : 'current' },
        dry_run: dryRun,
      }),
    })
    const body = await response.text()
    setActionResult(body)
    if (response.ok && !dryRun) await refresh()
  }

  const flush = async () => {
    const response = await fetch(endpoint('flush'), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ selector: {}, confirm }),
    })
    const body = await response.text()
    setActionResult(body)
    if (response.ok) {
      setConfirm('')
      await refresh()
    }
  }

  const cards = [
    ['Health', health],
    ['Backend', capabilities?.backend ?? '—'],
    ['Hit rate', `${hitRate.toFixed(1)}%`],
    ['Entries', String(stats?.total_entries ?? 0)],
    ['L1 / L2 hits', `${stats?.l1_hit_count ?? 0} / ${stats?.l2_hit_count ?? 0}`],
    ['Exact / semantic', `${stats?.exact_hit_count ?? 0} / ${stats?.semantic_hit_count ?? 0}`],
    ['Singleflight waiters', String(stats?.singleflight_waiters ?? 0)],
    ['Fail-open / stale', `${stats?.fail_open_count ?? 0} / ${stats?.stale_miss_count ?? 0}`],
  ]

  return (
    <main className={styles.page}>
      <header className={styles.header}>
        <div>
          <h1>Response Cache</h1>
          <p>Health, backend capabilities, cache tiers, freshness, and scoped invalidation.</p>
        </div>
        <button onClick={() => void refresh()}>Refresh</button>
      </header>
      {error && <div className={styles.error}>{error}</div>}
      <section className={styles.grid}>
        {cards.map(([label, value]) => (
          <article className={styles.card} key={label}>
            <span>{label}</span>
            <strong>{value}</strong>
          </article>
        ))}
      </section>
      <section className={styles.panel}>
        <h2>Capabilities</h2>
        <pre>{JSON.stringify(capabilities, null, 2)}</pre>
      </section>
      <section className={styles.panel}>
        <h2>Scoped invalidation</h2>
        <div className={styles.formRow}>
          <input
            value={recipe}
            onChange={(event) => setRecipe(event.target.value)}
            placeholder="Recipe"
          />
          <input
            value={model}
            onChange={(event) => setModel(event.target.value)}
            placeholder="Request model"
          />
          <input
            value={protocol}
            onChange={(event) => setProtocol(event.target.value)}
            placeholder="Protocol"
          />
          <label>
            <input
              type="checkbox"
              checked={dryRun}
              onChange={(event) => setDryRun(event.target.checked)}
            />
            Dry run
          </label>
          <button disabled={isReadonly} onClick={() => void invalidate()}>
            Invalidate
          </button>
        </div>
      </section>
      <section className={styles.panel}>
        <h2>Flush by epoch</h2>
        <p>This advances the cache epoch; it never runs backend-wide FLUSHALL.</p>
        <div className={styles.formRow}>
          <input
            value={confirm}
            onChange={(event) => setConfirm(event.target.value)}
            placeholder='Type "flush response cache"'
          />
          <button
            className={styles.danger}
            disabled={isReadonly || confirm !== 'flush response cache'}
            onClick={() => void flush()}
          >
            Flush
          </button>
        </div>
      </section>
      {actionResult && (
        <section className={styles.panel}>
          <h2>Last operation</h2>
          <pre>{actionResult}</pre>
        </section>
      )}
      <section className={styles.panel}>
        <h2>Audit</h2>
        <p>Invalidate and flush actions are recorded by the immutable management audit pipeline.</p>
        <pre>{JSON.stringify(auditEntries.slice(-20).reverse(), null, 2)}</pre>
      </section>
    </main>
  )
}

export default ResponseCachePage
