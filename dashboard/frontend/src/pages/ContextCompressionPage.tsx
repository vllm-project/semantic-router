import React, { useCallback, useEffect, useMemo, useState } from 'react'
import { useAuth } from '../contexts/AuthContext'
import { useReadonly } from '../contexts/ReadonlyContext'
import { canWriteConfig } from '../utils/accessControl'
import styles from './ContextCompressionPage.module.css'

interface CompressionCapabilities {
  strategies: string[]
  scoring_methods: string[]
  targets: string[]
  recovery: boolean
  json_structure_preserved: boolean
  multimodal_preserved: boolean
}

interface CompressionStats {
  requests: number
  applied: number
  skipped: number
  failures: number
  recovery_writes: number
  recovery_reads: number
  recovery_failures: number
  tokens_before: number
  tokens_after: number
  tokens_saved: number
  estimated_cost_saved_usd: number
  audit?: Array<{
    sequence: number
    action: string
    role: string
    status: number
    timestamp: string
    hash: string
  }>
}

interface CompressionPreview {
  valid: boolean
  applied: boolean
  tokens_before: number
  tokens_after: number
  tokens_saved: number
  target_request_tokens: number
  available_input_tokens: number
  reserved_output_tokens: number
  messages_compressed: number
  blocks_compressed: number
  omitted_chunks: number
  recovery_keys: number
  token_counter_source: string
  trigger_reason?: string
  skip_reason?: string
  warnings?: string[]
}

const endpoint = (path: string) => `/api/router/api/v1/context-compression/${path}`

const defaultConfiguration = JSON.stringify(
  {
    enabled: true,
    mode: 'auto',
    budget: {
      trigger_tokens: 'auto',
      target_tokens: 'auto',
      reserve_output_tokens: 4096,
    },
    targets: {
      tool_outputs: { mode: 'extractive', min_tokens: 2000, target_tokens: 1000 },
      history: { mode: 'preserve' },
      rag: { mode: 'preserve' },
      memory: { mode: 'preserve' },
    },
    scoring: { method: 'bm25' },
    failure_mode: 'fail_open',
  },
  null,
  2,
)

const defaultRequest = JSON.stringify(
  {
    model: 'vllm-sr/auto',
    messages: [
      { role: 'user', content: 'Summarize the authentication failure.' },
      {
        role: 'tool',
        tool_call_id: 'call_logs',
        content: 'Paste a representative large tool output here.',
      },
    ],
  },
  null,
  2,
)

interface ContextCompressionPageProps {
  embedded?: boolean
}

const ContextCompressionPage: React.FC<ContextCompressionPageProps> = ({ embedded = false }) => {
  const { isReadonly } = useReadonly()
  const { user } = useAuth()
  const canManage = !isReadonly && canWriteConfig(user)
  const [capabilities, setCapabilities] = useState<CompressionCapabilities | null>(null)
  const [stats, setStats] = useState<CompressionStats | null>(null)
  const [health, setHealth] = useState('loading')
  const [error, setError] = useState<string | null>(null)
  const [configuration, setConfiguration] = useState(defaultConfiguration)
  const [requestBody, setRequestBody] = useState(defaultRequest)
  const [model, setModel] = useState('vllm-sr/auto')
  const [contextWindow, setContextWindow] = useState('32768')
  const [preview, setPreview] = useState<CompressionPreview | null>(null)
  const [recoveryScope, setRecoveryScope] = useState({
    recipe: '',
    decision: '',
    user_id: '',
    request_id: '',
  })

  const refresh = useCallback(async () => {
    try {
      const [capabilityResponse, healthResponse, statsResponse] = await Promise.all([
        fetch(endpoint('capabilities')),
        fetch(endpoint('health')),
        fetch(endpoint('stats')),
      ])
      if (!capabilityResponse.ok || !statsResponse.ok) {
        throw new Error('Context compression management API is unavailable')
      }
      setCapabilities((await capabilityResponse.json()) as CompressionCapabilities)
      setStats((await statsResponse.json()) as CompressionStats)
      setHealth(healthResponse.ok ? 'healthy' : 'unhealthy')
      setError(null)
    } catch (requestError) {
      setHealth('unavailable')
      setError(
        requestError instanceof Error
          ? requestError.message
          : 'Unable to load context compression status',
      )
    }
  }, [])

  useEffect(() => {
    void refresh()
  }, [refresh])

  const savingsRatio = useMemo(() => {
    if (!stats || stats.tokens_before <= 0) return 0
    return (stats.tokens_saved / stats.tokens_before) * 100
  }, [stats])

  const runPreview = async () => {
    try {
      const response = await fetch(endpoint('preview'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          configuration: JSON.parse(configuration),
          request_body: JSON.parse(requestBody),
          model,
          context_window: Number(contextWindow) || 0,
        }),
      })
      const payload = await response.json()
      if (!response.ok) {
        throw new Error(payload?.error?.message ?? 'Preview failed')
      }
      setPreview(payload as CompressionPreview)
      setError(null)
    } catch (previewError) {
      setError(previewError instanceof Error ? previewError.message : 'Preview failed')
    }
  }

  const invalidateRecovery = async () => {
    try {
      const response = await fetch(endpoint('recovery/invalidate'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(recoveryScope),
      })
      const payload = await response.json()
      if (!response.ok) {
        throw new Error(payload?.error?.message ?? 'Recovery invalidation failed')
      }
      setError(null)
      await refresh()
    } catch (invalidateError) {
      setError(
        invalidateError instanceof Error ? invalidateError.message : 'Recovery invalidation failed',
      )
    }
  }

  const cards = [
    ['Health', health],
    ['Runtime requests', String(stats?.requests ?? 0)],
    ['Applied / skipped', `${stats?.applied ?? 0} / ${stats?.skipped ?? 0}`],
    ['Tokens saved', String(stats?.tokens_saved ?? 0)],
    ['Estimated cost saved', `$${(stats?.estimated_cost_saved_usd ?? 0).toFixed(4)}`],
    ['Savings ratio', `${savingsRatio.toFixed(1)}%`],
    ['Recovery writes', String(stats?.recovery_writes ?? 0)],
    [
      'Recovery reads / failures',
      `${stats?.recovery_reads ?? 0} / ${stats?.recovery_failures ?? 0}`,
    ],
  ]

  return (
    <main className={`${styles.page} ${embedded ? styles.embedded : ''}`}>
      <header className={styles.sectionHeader}>
        <div>
          <span className={styles.eyebrow}>Plugin health and controls</span>
          <h2>Context Compression</h2>
          <p>
            Provider-aware budgets, target policies, recovery capability, and redacted previews.
          </p>
        </div>
        <button className={styles.refreshButton} onClick={() => void refresh()}>
          Refresh
        </button>
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
        <h2>Redacted preview</h2>
        <p>
          The API returns plans and token counts only. Omitted source content is never returned.
        </p>
        <div className={styles.formRow}>
          <label>
            Model
            <input value={model} onChange={(event) => setModel(event.target.value)} />
          </label>
          <label>
            Context window
            <input
              value={contextWindow}
              onChange={(event) => setContextWindow(event.target.value)}
            />
          </label>
          <button disabled={isReadonly} onClick={() => void runPreview()}>
            Preview
          </button>
        </div>
        <label>
          Compression configuration
          <textarea
            rows={18}
            value={configuration}
            onChange={(event) => setConfiguration(event.target.value)}
          />
        </label>
        <label>
          Sample request
          <textarea
            rows={18}
            value={requestBody}
            onChange={(event) => setRequestBody(event.target.value)}
          />
        </label>
      </section>
      {preview && (
        <section className={styles.panel}>
          <h2>Preview result</h2>
          <pre>{JSON.stringify(preview, null, 2)}</pre>
        </section>
      )}
      <section className={styles.panel}>
        <h2>Preview audit</h2>
        <pre>{JSON.stringify(stats?.audit ?? [], null, 2)}</pre>
      </section>
      <section className={styles.panel}>
        <h2>Invalidate recovery scope</h2>
        <p>Requires compression.manage. Scope values are hashed and are not returned.</p>
        <div className={styles.formRow}>
          {Object.entries(recoveryScope).map(([key, value]) => (
            <label key={key}>
              {key}
              <input
                value={value}
                onChange={(event) =>
                  setRecoveryScope((current) => ({ ...current, [key]: event.target.value }))
                }
              />
            </label>
          ))}
          <button
            disabled={!canManage || Object.values(recoveryScope).some((value) => !value.trim())}
            onClick={() => void invalidateRecovery()}
          >
            Invalidate
          </button>
        </div>
      </section>
    </main>
  )
}

export default ContextCompressionPage
