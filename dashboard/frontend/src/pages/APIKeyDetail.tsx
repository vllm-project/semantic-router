import { useEffect, useMemo, useState } from 'react'
import useAccessibleDialog from '../hooks/useAccessibleDialog'
import { copyText } from '../utils/clipboard'
import {
  inferenceAccessApi,
  type AccessAPIKey,
  type AccessBudget,
  type AccessGroup,
  type AccessTeam,
  type AccessUser,
} from '../utils/inferenceAccessApi'
import {
  EMPTY_USAGE,
  effectivePatterns,
  formatDate,
  formatNumber,
  ownerLabel,
} from './AccessControlDetailSupport'
import styles from './AccessControlPage.module.css'

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
}: KeyDetailProps) {
  const [key, setKey] = useState<AccessAPIKey | null>(null)
  const [usage, setUsage] = useState(EMPTY_USAGE)
  const [secret, setSecret] = useState('')
  const [secretVisible, setSecretVisible] = useState(false)
  const [loading, setLoading] = useState(true)
  const [pending, setPending] = useState(false)
  const [error, setError] = useState('')
  const [copied, setCopied] = useState('')
  const [snippet, setSnippet] = useState<'python' | 'javascript' | 'curl'>('python')
  const [rotateArmed, setRotateArmed] = useState(false)
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
      ? inferenceAccessApi.selfUsage({ keyId, from })
      : inferenceAccessApi.usage({ keyId, from })
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
    if (!copied) return
    const timer = window.setTimeout(() => setCopied(''), 1800)
    return () => window.clearTimeout(timer)
  }, [copied])

  const patterns = useMemo(() => (key ? effectivePatterns(key, groups) : []), [groups, key])
  const effectiveBudget = key
    ? budgets.find((budget) => budget.id === (key.effectiveBudgetId || key.budgetId))
    : undefined
  const model = patterns.find((pattern) => !pattern.includes('*')) || 'vllm-sr/mom-v1-lite'
  const baseURL = `${window.location.origin}/v1`
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

  const rotate = async () => {
    setPending(true)
    setError('')
    try {
      const next = selfService
        ? await inferenceAccessApi.rotateSelfKey(keyId)
        : await inferenceAccessApi.rotateKey(keyId)
      setKey(next)
      setSecret(next.secret)
      setSecretVisible(true)
      setRotateArmed(false)
      onChanged()
    } catch (nextError) {
      setError(nextError instanceof Error ? nextError.message : 'Could not rotate API key')
    } finally {
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
            ×
          </button>
        </header>
        <div className={styles.detailBody}>
          {loading ? <div className={styles.detailLoading}>Loading key details…</div> : null}
          {error ? (
            <div className={styles.modalError} role="alert">
              <span>!</span>
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
                  <span>P95 latency</span>
                  <strong>{formatNumber(usage.p95LatencyMs)} ms</strong>
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
                      {effectiveBudget
                        ? `${effectiveBudget.name} · ${key.budgetPolicySource || 'key'} policy`
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
                      {patterns.length
                        ? patterns.map((pattern) => <code key={pattern}>{pattern}</code>)
                        : 'No models assigned'}
                    </dd>
                  </div>
                  <div>
                    <dt>Model policy</dt>
                    <dd>
                      {key.modelPolicySource ? `${key.modelPolicySource} policy` : 'Not assigned'}
                    </dd>
                  </div>
                  <div>
                    <dt>RPM</dt>
                    <dd>{effectiveBudget ? formatNumber(effectiveBudget.rpm) : 'Unlimited'}</dd>
                  </div>
                  <div>
                    <dt>TPM</dt>
                    <dd>{effectiveBudget ? formatNumber(effectiveBudget.tpm) : 'Unlimited'}</dd>
                  </div>
                  <div>
                    <dt>Daily tokens</dt>
                    <dd>
                      {effectiveBudget ? formatNumber(effectiveBudget.dailyTokens) : 'Unlimited'}
                    </dd>
                  </div>
                  <div>
                    <dt>Expires</dt>
                    <dd>{formatDate(key.expiresAt)}</dd>
                  </div>
                </dl>
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
            {rotateArmed ? (
              <div className={styles.detailConfirm} role="alert">
                <span>The current secret will stop working.</span>
                <button type="button" onClick={() => setRotateArmed(false)} disabled={pending}>
                  Cancel
                </button>
                <button type="button" onClick={() => void rotate()} disabled={pending}>
                  {pending ? 'Rotating…' : 'Confirm rotation'}
                </button>
              </div>
            ) : null}
            {canEditPolicy ? (
              <button type="button" className={styles.secondaryButton} onClick={() => onEdit(key)}>
                Edit policy
              </button>
            ) : null}
            <button
              type="button"
              className={styles.secondaryButton}
              onClick={() => void toggle()}
              disabled={pending}
            >
              {key.status === 'active' ? 'Revoke' : 'Activate'}
            </button>
            <button
              type="button"
              className={styles.secondaryButton}
              onClick={() => setRotateArmed(true)}
              disabled={pending}
            >
              Rotate key
            </button>
            <button type="button" className={styles.primaryButton} onClick={onClose}>
              Done
            </button>
          </footer>
        ) : null}
      </aside>
    </div>
  )
}
