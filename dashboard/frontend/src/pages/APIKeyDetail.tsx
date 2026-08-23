import { useEffect, useMemo, useState } from 'react'
import ProductIcon from '../components/ProductIcon'
import { useReadonly } from '../contexts/ReadonlyContext'
import useAccessibleDialog from '../hooks/useAccessibleDialog'
import { copyText } from '../utils/clipboard'
import {
  inferenceAccessApi,
  type AccessAPIKey,
  type AccessBudget,
  type AccessGroup,
  type AccessTeam,
  type AccessUser,
  type QuotaMeter,
} from '../utils/inferenceAccessApi'
import { routerPublicEndpoint } from '../utils/routerPublicApi'
import {
  EMPTY_USAGE,
  costCoverageLabel,
  effectiveResources,
  formatCosts,
  formatDate,
  formatNumber,
  formatQuotaValue,
  ownerLabel,
  quotaMeterLabel,
  quotaProgress,
  quotaResetLabel,
} from './AccessControlDetailSupport'
import styles from './AccessControlPage.module.css'

const KEY_QUOTA_REFRESH_MS = 5000

interface KeyDetailProps {
  keyId: string
  users: AccessUser[]
  teams: AccessTeam[]
  groups: AccessGroup[]
  budgets: AccessBudget[]
  canManage: boolean
  canEditPolicy: boolean
  selfService?: boolean
  onEdit: (key: AccessAPIKey) => void
  onClose: () => void
  onChanged: () => void
  onDeleted: () => void
}

export function APIKeyDetail({
  keyId,
  users,
  teams,
  groups,
  budgets,
  canManage,
  canEditPolicy,
  selfService = false,
  onEdit,
  onClose,
  onChanged,
  onDeleted,
}: KeyDetailProps) {
  const { routerPublicUrl } = useReadonly()
  const [key, setKey] = useState<AccessAPIKey | null>(null)
  const [usage, setUsage] = useState(EMPTY_USAGE)
  const [secret, setSecret] = useState('')
  const [secretVisible, setSecretVisible] = useState(false)
  const [loading, setLoading] = useState(true)
  const [pending, setPending] = useState(false)
  const [error, setError] = useState('')
  const [copied, setCopied] = useState('')
  const [snippet, setSnippet] = useState<'python' | 'javascript' | 'curl'>('python')
  const [renewArmed, setRenewArmed] = useState(false)
  const [deleteArmed, setDeleteArmed] = useState(false)
  const dialogRef = useAccessibleDialog<HTMLDivElement>({
    isOpen: true,
    onClose,
    dismissible: !pending,
  })

  useEffect(() => {
    setLoading(true)
    setError('')
    const from = new Date(Date.now() - 7 * 24 * 60 * 60 * 1000).toISOString()
    const keyRequest = selfService
      ? inferenceAccessApi.selfKey(keyId)
      : inferenceAccessApi.key(keyId)
    const usageRequest = selfService
      ? inferenceAccessApi.selfKeyUsage(keyId, { from })
      : inferenceAccessApi.keyUsage(keyId, { from })
    void Promise.all([keyRequest, usageRequest])
      .then(([nextKey, nextUsage]) => {
        setKey(nextKey)
        setUsage(nextUsage)
      })
      .catch((nextError) =>
        setError(nextError instanceof Error ? nextError.message : 'Could not load API key'),
      )
      .finally(() => setLoading(false))
  }, [keyId, selfService])

  useEffect(() => {
    let cancelled = false
    let inFlight = false
    const refreshQuota = async () => {
      if (document.hidden || inFlight) return
      inFlight = true
      try {
        const next = selfService
          ? await inferenceAccessApi.selfKey(keyId)
          : await inferenceAccessApi.key(keyId)
        if (!cancelled) setKey(next)
      } catch {
        // Keep the last complete snapshot during a transient background refresh failure.
      } finally {
        inFlight = false
      }
    }
    const refreshWhenVisible = () => {
      if (!document.hidden) void refreshQuota()
    }
    const interval = window.setInterval(() => void refreshQuota(), KEY_QUOTA_REFRESH_MS)
    document.addEventListener('visibilitychange', refreshWhenVisible)
    return () => {
      cancelled = true
      window.clearInterval(interval)
      document.removeEventListener('visibilitychange', refreshWhenVisible)
    }
  }, [keyId, selfService])

  useEffect(() => {
    if (!copied) return
    const timer = window.setTimeout(() => setCopied(''), 1800)
    return () => window.clearTimeout(timer)
  }, [copied])

  const resources = useMemo(() => (key ? effectiveResources(key, groups) : []), [groups, key])
  const effectiveBudget = key
    ? budgets.find((budget) => budget.id === (key.effectiveBudgetId || key.budgetId))
    : undefined
  const quotaBudget = key?.quota
    ? budgets.find((budget) => budget.id === key.quota?.budgetId) || effectiveBudget
    : effectiveBudget
  const quotaSource = key?.quota?.source || key?.budgetPolicySource
  const quotaSourceLabel =
    quotaSource === 'key'
      ? 'Key limit'
      : quotaSource === 'user'
        ? 'User default'
        : quotaSource === 'team'
          ? 'Team default'
          : ''
  const model =
    resources.find((resource) => resource.resourceType === 'entrypoint')?.resourceId ||
    resources[0]?.resourceId ||
    'YOUR_MODEL'
  // Inference is served by the Router's public listener, never by the Dashboard.
  const baseURL = routerPublicEndpoint(routerPublicUrl, '/v1')
  const snippets = useMemo(
    () => ({
      python: `import os\nfrom openai import OpenAI\n\nclient = OpenAI(\n    base_url="${baseURL}",\n    api_key=${secret ? `"${secret}"` : 'os.environ["VLLM_SR_API_KEY"]'},\n)\n\nresponse = client.chat.completions.create(\n    model="${model}",\n    messages=[{"role": "user", "content": "Hello"}],\n)\nprint(response.choices[0].message.content)`,
      javascript: `import OpenAI from "openai";\n\nconst client = new OpenAI({\n  baseURL: "${baseURL}",\n  apiKey: ${secret ? `"${secret}"` : 'process.env.VLLM_SR_API_KEY'},\n});\n\nconst response = await client.chat.completions.create({\n  model: "${model}",\n  messages: [{ role: "user", content: "Hello" }],\n});\nconsole.log(response.choices[0].message.content);`,
      curl: `curl ${baseURL}/chat/completions \\\n  -H "Authorization: Bearer ${secret || '$VLLM_SR_API_KEY'}" \\\n  -H "Content-Type: application/json" \\\n  -d '{"model":"${model}","messages":[{"role":"user","content":"Hello"}]}'`,
    }),
    [baseURL, model, secret],
  )

  const reveal = async () => {
    setPending(true)
    setError('')
    try {
      const result = selfService
        ? await inferenceAccessApi.selfKeySecret(keyId)
        : await inferenceAccessApi.keySecret(keyId)
      setSecret(result.secret)
      setSecretVisible(true)
    } catch (nextError) {
      setError(nextError instanceof Error ? nextError.message : 'Could not reveal API key')
    } finally {
      setPending(false)
    }
  }

  const renew = async () => {
    setPending(true)
    setError('')
    try {
      const next = selfService
        ? await inferenceAccessApi.rotateSelfKey(keyId)
        : await inferenceAccessApi.rotateKey(keyId)
      setKey(next)
      setSecret(next.secret)
      setSecretVisible(true)
      setRenewArmed(false)
      onChanged()
    } catch (nextError) {
      setError(nextError instanceof Error ? nextError.message : 'Could not renew API key')
    } finally {
      setPending(false)
    }
  }

  const remove = async () => {
    setPending(true)
    setError('')
    try {
      if (selfService) await inferenceAccessApi.deleteSelfKey(keyId)
      else await inferenceAccessApi.deleteKey(keyId)
      onDeleted()
    } catch (nextError) {
      setError(nextError instanceof Error ? nextError.message : 'Could not delete API key')
      setDeleteArmed(false)
      setPending(false)
    }
  }

  const toggle = async () => {
    if (!key) return
    setPending(true)
    setError('')
    try {
      const status = key.status === 'active' ? 'disabled' : 'active'
      const next = selfService
        ? await inferenceAccessApi.setSelfKeyStatus(key.id, status)
        : await inferenceAccessApi.setKeyStatus(key.id, status)
      setKey((current) => (current ? { ...current, status: next.status } : current))
      onChanged()
    } catch (nextError) {
      setError(nextError instanceof Error ? nextError.message : 'Could not update API key')
    } finally {
      setPending(false)
    }
  }

  const copy = async (value: string, label: string) => {
    if (await copyText(value)) setCopied(label)
  }

  return (
    <div
      className={styles.detailBackdrop}
      onMouseDown={(event) => {
        if (event.target === event.currentTarget && !pending) onClose()
      }}
    >
      <aside
        ref={dialogRef}
        className={styles.detailDrawer}
        role="dialog"
        aria-modal="true"
        aria-labelledby="key-detail-title"
        aria-busy={loading || pending}
        tabIndex={-1}
      >
        <header className={styles.detailHeader}>
          <div className={styles.detailHeaderIdentity}>
            <div className={styles.modalLogo} aria-hidden="true">
              <img src="/vllm.png" alt="" />
            </div>
            <div>
              <span>API Key</span>
              <h2 id="key-detail-title">{key?.name || 'Key details'}</h2>
              <p>
                {key
                  ? `${key.prefix}•••••••• · ${ownerLabel(key, users, teams)}`
                  : 'Loading key details…'}
              </p>
            </div>
          </div>
          <button type="button" className={styles.modalClose} onClick={onClose} aria-label="Close">
            <ProductIcon name="close" />
          </button>
        </header>
        <div className={styles.detailBody}>
          {loading ? <div className={styles.detailLoading}>Loading key details…</div> : null}
          {error ? (
            <div className={styles.modalError} role="alert">
              <ProductIcon name="alert" aria-hidden="true" />
              <div>
                <strong>Couldn’t continue</strong>
                <p>{error}</p>
              </div>
            </div>
          ) : null}
          {key ? (
            <>
              <section className={styles.secretPanel}>
                <div className={styles.secretPanelHeading}>
                  <div>
                    <span>Secret</span>
                    <strong>{secret ? 'Original key' : 'Hidden by default'}</strong>
                  </div>
                  {canManage ? (
                    <button
                      type="button"
                      className={styles.secondaryButton}
                      onClick={() => (secret ? setSecretVisible((value) => !value) : void reveal())}
                      disabled={pending}
                    >
                      <ProductIcon name="key" />
                      {secret ? (secretVisible ? 'Hide' : 'Show') : 'Reveal key'}
                    </button>
                  ) : null}
                </div>
                <div className={styles.secretInline}>
                  <code>{secret && secretVisible ? secret : `${key.prefix}${'•'.repeat(28)}`}</code>
                  <button
                    type="button"
                    disabled={!secret}
                    onClick={() => void copy(secret, 'secret')}
                  >
                    <ProductIcon name="copy" />
                    {copied === 'secret' ? 'Copied' : 'Copy'}
                  </button>
                </div>
              </section>

              <div className={styles.detailMetrics}>
                <article>
                  <span>7-day requests</span>
                  <strong>{formatNumber(usage.requests)}</strong>
                </article>
                <article>
                  <span>7-day tokens</span>
                  <strong>{formatNumber(usage.totalTokens)}</strong>
                </article>
                <article>
                  <span>Success rate</span>
                  <strong>
                    {usage.requests
                      ? `${((usage.successful / usage.requests) * 100).toFixed(1)}%`
                      : '—'}
                  </strong>
                </article>
                <article>
                  <span>7-day spend</span>
                  <strong title={formatCosts(usage.costs)}>{formatCosts(usage.costs)}</strong>
                  <small>{costCoverageLabel(usage.costs)}</small>
                </article>
              </div>

              <section className={styles.detailSection}>
                <div className={styles.detailSectionHeading}>
                  <span>Access</span>
                  <h3>Models & quota</h3>
                </div>
                <dl className={styles.detailGrid}>
                  <div>
                    <dt>Status</dt>
                    <dd>
                      <span
                        className={
                          key.status === 'active' ? styles.detailStatusLive : styles.detailStatusOff
                        }
                      >
                        {key.status}
                      </span>
                    </dd>
                  </div>
                  <div>
                    <dt>Owner</dt>
                    <dd>{ownerLabel(key, users, teams)}</dd>
                  </div>
                  <div>
                    <dt>Budget</dt>
                    <dd>
                      {key.quota
                        ? `${quotaBudget?.name || key.quota.budgetName} · ${quotaSourceLabel}`
                        : effectiveBudget
                          ? `${effectiveBudget.name} · ${quotaSourceLabel}`
                          : 'No quota policy'}
                    </dd>
                  </div>
                  <div>
                    <dt>Team context</dt>
                    <dd>
                      {key.contextTeamId
                        ? teams.find((team) => team.id === key.contextTeamId)?.name ||
                          key.contextTeamId
                        : 'Personal policy'}
                    </dd>
                  </div>
                  <div>
                    <dt>Created</dt>
                    <dd>{formatDate(key.createdAt)}</dd>
                  </div>
                  <div>
                    <dt>Last used</dt>
                    <dd>{formatDate(key.lastUsedAt)}</dd>
                  </div>
                  <div className={styles.detailGridWide}>
                    <dt>Visible models</dt>
                    <dd className={styles.detailTags}>
                      {resources.length
                        ? resources.map((resource) => (
                            <code key={`${resource.resourceType}:${resource.resourceId}`}>
                              {resource.resourceId}
                            </code>
                          ))
                        : 'No models assigned'}
                    </dd>
                  </div>
                  <div>
                    <dt>Model policy</dt>
                    <dd>
                      {key.accessPolicySources?.length
                        ? `${key.accessPolicySources.join(' + ')} policy`
                        : 'Not assigned'}
                    </dd>
                  </div>
                  <div>
                    <dt>Expires</dt>
                    <dd>{formatDate(key.expiresAt)}</dd>
                  </div>
                </dl>
                {key.quota ? (
                  <div className={styles.quotaGrid}>
                    {key.quota.meters.map((meter: QuotaMeter) => {
                      const percent = quotaProgress(meter)
                      const resetLabel = quotaResetLabel(meter)
                      return (
                        <article key={`${meter.policyId}:${meter.ruleId}:${meter.bindingId}`}>
                          <div>
                            <span>{quotaMeterLabel(meter)}</span>
                            <small>
                              {formatQuotaValue(meter, meter.used)} of{' '}
                              {formatQuotaValue(meter, meter.limit)}
                            </small>
                          </div>
                          <strong>
                            {meter.remaining === null
                              ? 'Usage unavailable'
                              : `${formatQuotaValue(meter, meter.remaining)} left`}
                          </strong>
                          <div className={styles.quotaTrack} aria-hidden="true">
                            <i style={{ width: `${percent}%` }} />
                          </div>
                          {resetLabel ? (
                            <small className={styles.quotaReset}>{resetLabel}</small>
                          ) : null}
                          {meter.completeness !== 'complete' ? (
                            <small className={styles.quotaReset}>
                              Usage data is {meter.completeness}.
                            </small>
                          ) : null}
                        </article>
                      )
                    })}
                  </div>
                ) : null}
              </section>

              <section className={styles.detailSection}>
                <div className={styles.detailSectionHeading}>
                  <span>Quickstart</span>
                  <h3>Send your first request</h3>
                </div>
                <div className={styles.codeTabs}>
                  {(['python', 'javascript', 'curl'] as const).map((item) => (
                    <button
                      type="button"
                      key={item}
                      className={snippet === item ? styles.codeTabActive : ''}
                      onClick={() => setSnippet(item)}
                    >
                      {item === 'javascript' ? 'JavaScript' : item === 'python' ? 'Python' : 'cURL'}
                    </button>
                  ))}
                  <button
                    type="button"
                    className={styles.codeCopy}
                    onClick={() => void copy(snippets[snippet], 'snippet')}
                  >
                    <ProductIcon name="copy" />
                    {copied === 'snippet' ? 'Copied' : 'Copy code'}
                  </button>
                </div>
                <pre className={styles.codeBlock}>
                  <code>{snippets[snippet]}</code>
                </pre>
                {!secret ? (
                  <p className={styles.codeHint}>
                    Reveal the key to place it directly in the sample. Environment variables remain
                    the safer default.
                  </p>
                ) : null}
              </section>
            </>
          ) : null}
        </div>
        {key && canManage ? (
          <footer className={styles.detailFooter}>
            {renewArmed ? (
              <div className={styles.detailConfirm} role="alert">
                <span>Renewing replaces the current secret.</span>
                <button type="button" onClick={() => setRenewArmed(false)} disabled={pending}>
                  Cancel
                </button>
                <button type="button" onClick={() => void renew()} disabled={pending}>
                  {pending ? 'Renewing…' : 'Renew key'}
                </button>
              </div>
            ) : null}
            {deleteArmed ? (
              <div className={styles.detailConfirm} role="alert">
                <span>Delete this key permanently?</span>
                <button type="button" onClick={() => setDeleteArmed(false)} disabled={pending}>
                  Cancel
                </button>
                <button type="button" onClick={() => void remove()} disabled={pending}>
                  <ProductIcon name="trash" />
                  {pending ? 'Deleting…' : 'Delete key'}
                </button>
              </div>
            ) : null}
            {canEditPolicy ? (
              <button type="button" className={styles.secondaryButton} onClick={() => onEdit(key)}>
                <ProductIcon name="edit" /> Edit access & quota
              </button>
            ) : null}
            <button
              type="button"
              className={styles.secondaryButton}
              onClick={() => void toggle()}
              disabled={pending}
            >
              <ProductIcon name="power" />
              {key.status === 'active' ? 'Disable' : 'Enable'}
            </button>
            <button
              type="button"
              className={styles.secondaryButton}
              onClick={() => {
                setDeleteArmed(false)
                setRenewArmed(true)
              }}
              disabled={pending}
            >
              <ProductIcon name="refresh" /> Renew key
            </button>
            <button
              type="button"
              className={styles.dangerButton}
              onClick={() => {
                setRenewArmed(false)
                setDeleteArmed(true)
              }}
              disabled={pending}
            >
              <ProductIcon name="trash" /> Delete
            </button>
            <button type="button" className={styles.primaryButton} onClick={onClose}>
              <ProductIcon name="check" /> Done
            </button>
          </footer>
        ) : null}
      </aside>
    </div>
  )
}
