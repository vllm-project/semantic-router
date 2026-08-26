import { useEffect, useMemo, useState } from 'react'
import ProductIcon from '../components/ProductIcon'
import ProductLoadingState from '../components/ProductLoadingState'
import { useReadonly } from '../contexts/ReadonlyContext'
import useAccessibleDialog from '../hooks/useAccessibleDialog'
import { copyText } from '../utils/clipboard'
import {
  inferenceAccessApi,
  type AccessAPIKey,
  type AccessBudget,
  type AccessGroup,
  type QuotaMeter,
} from '../utils/inferenceAccessApi'
import {
  fetchKeyScopedRoutingCatalog,
  type KeyScopedRoutingCatalog,
} from '../utils/keyScopedRoutingCatalog'
import { routerPublicEndpoint } from '../utils/routerPublicApi'
import {
  EMPTY_USAGE,
  costCoverageLabel,
  formatCosts,
  formatDate,
  formatNumber,
  formatQuotaValue,
  quotaMeterLabel,
  quotaProgress,
  quotaResetLabel,
} from './AccessControlDetailSupport'
import {
  apiKeyQuickstartModel,
  apiKeyResourceResolutions,
  apiKeyVisibleResourceNames,
} from './apiKeyResourceNames'
import { buildAPIKeyQuickstartSnippets } from './apiKeyQuickstartSnippets'
import styles from './AccessControlPage.module.css'

const KEY_QUOTA_REFRESH_MS = 5000

interface KeyDetailProps {
  keyId: string
  canManage: boolean
  canReveal: boolean
  canEditPolicy: boolean
  selfService?: boolean
  selfUserId: string
  onEdit: (key: AccessAPIKey) => void
  onClose: () => void
  onChanged: () => void
  onDeleted: () => void
}

export function APIKeyDetail({
  keyId,
  canManage,
  canReveal,
  canEditPolicy,
  selfService = false,
  selfUserId,
  onEdit,
  onClose,
  onChanged,
  onDeleted,
}: KeyDetailProps) {
  const { routerPublicUrl } = useReadonly()
  const [key, setKey] = useState<AccessAPIKey | null>(null)
  const [ownerName, setOwnerName] = useState('')
  const [contextTeamName, setContextTeamName] = useState('')
  const [assignedGroups, setAssignedGroups] = useState<AccessGroup[]>([])
  const [assignedBudget, setAssignedBudget] = useState<AccessBudget | null>(null)
  const [routingCatalog, setRoutingCatalog] = useState<KeyScopedRoutingCatalog | null>()
  const [routingCatalogAttempt, setRoutingCatalogAttempt] = useState(0)
  const [canSelfManage, setCanSelfManage] = useState(false)
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
  const relationshipKeyId = key?.id
  const relationshipOwnerType = key?.ownerType
  const relationshipOwnerId = key?.ownerId
  const relationshipContextTeamId = key?.contextTeamId
  const accessPolicyKey = key?.accessGroupIds.join(',') ?? ''
  const relationshipBudgetId = key?.effectiveBudgetId || key?.budgetId || key?.quota?.budgetId
  const relationshipKey = useMemo(
    () =>
      relationshipKeyId && relationshipOwnerType && relationshipOwnerId
        ? {
            id: relationshipKeyId,
            ownerType: relationshipOwnerType,
            ownerId: relationshipOwnerId,
            contextTeamId: relationshipContextTeamId,
            accessGroupIds: accessPolicyKey ? accessPolicyKey.split(',') : [],
            budgetId: relationshipBudgetId,
          }
        : null,
    [
      accessPolicyKey,
      relationshipBudgetId,
      relationshipContextTeamId,
      relationshipKeyId,
      relationshipOwnerId,
      relationshipOwnerType,
    ],
  )
  const dialogRef = useAccessibleDialog<HTMLDivElement>({
    isOpen: true,
    onClose,
    dismissible: !pending,
  })

  useEffect(() => {
    let cancelled = false
    setLoading(true)
    setError('')
    setKey(null)
    const from = new Date(Date.now() - 7 * 24 * 60 * 60 * 1000).toISOString()
    const keyRequest = selfService
      ? inferenceAccessApi.selfKey(keyId)
      : inferenceAccessApi.key(keyId)
    const usageRequest = selfService
      ? inferenceAccessApi.selfKeyUsage(keyId, { from })
      : inferenceAccessApi.keyUsage(keyId, { from })
    setUsage(EMPTY_USAGE)
    void keyRequest
      .then((nextKey) => {
        if (!cancelled) setKey(nextKey)
      })
      .catch((nextError) => {
        if (!cancelled) {
          setError(nextError instanceof Error ? nextError.message : 'Could not load API key')
        }
      })
      .finally(() => {
        if (!cancelled) setLoading(false)
      })
    void usageRequest
      .then((nextUsage) => {
        if (!cancelled) setUsage(nextUsage)
      })
      .catch(() => {
        if (!cancelled) setUsage(EMPTY_USAGE)
      })
    return () => {
      cancelled = true
    }
  }, [keyId, selfService])

  useEffect(() => {
    const controller = new AbortController()
    setRoutingCatalog(undefined)
    void fetchKeyScopedRoutingCatalog(keyId, controller.signal)
      .then((catalog) => {
        if (controller.signal.aborted) return
        if (catalog.keyId !== keyId) {
          throw new Error('Router returned a routing catalog for a different API key.')
        }
        setRoutingCatalog(catalog)
      })
      .catch(() => {
        if (!controller.signal.aborted) setRoutingCatalog(null)
      })
    return () => controller.abort()
  }, [keyId, routingCatalogAttempt])

  useEffect(() => {
    let cancelled = false
    setOwnerName('')
    setContextTeamName('')
    setAssignedGroups([])
    setAssignedBudget(null)
    setCanSelfManage(false)
    if (!relationshipKey) return () => undefined

    const ownerRequest =
      relationshipKey.ownerType === 'user'
        ? inferenceAccessApi.userSummary(relationshipKey.ownerId).then((owner) => owner.name)
        : inferenceAccessApi.teamSummary(relationshipKey.ownerId).then((owner) => owner.name)
    const contextRequest = relationshipKey.contextTeamId
      ? inferenceAccessApi.teamSummary(relationshipKey.contextTeamId).then((team) => team.name)
      : Promise.resolve('')
    const groupsRequest = Promise.all(
      relationshipKey.accessGroupIds.map((policyId) =>
        inferenceAccessApi.groupSummary(policyId).catch(() => null),
      ),
    ).then((items) => items.filter((item): item is AccessGroup => item !== null))
    const budgetId = relationshipKey.budgetId
    const budgetRequest = budgetId
      ? inferenceAccessApi.budgetSummary(budgetId).catch(() => null)
      : Promise.resolve(null)
    const selfManageRequest = selfService
      ? relationshipKey.ownerType === 'user'
        ? Promise.resolve(relationshipKey.ownerId === selfUserId)
        : inferenceAccessApi
            .selfTeams()
            .then((catalog) =>
              catalog.items.some(
                (team) =>
                  team.id === relationshipKey.ownerId &&
                  team.members.some(
                    (membership) => membership.userId === selfUserId && membership.role === 'admin',
                  ),
              ),
            )
            .catch(() => false)
      : Promise.resolve(false)

    void Promise.all([
      ownerRequest.catch(() => relationshipKey.ownerId),
      contextRequest.catch(() => relationshipKey.contextTeamId || ''),
      groupsRequest,
      budgetRequest,
      selfManageRequest,
    ]).then(([nextOwner, nextContext, nextGroups, nextBudget, nextCanSelfManage]) => {
      if (cancelled) return
      setOwnerName(nextOwner)
      setContextTeamName(nextContext)
      setAssignedGroups(nextGroups)
      setAssignedBudget(nextBudget)
      setCanSelfManage(nextCanSelfManage)
    })
    return () => {
      cancelled = true
    }
  }, [relationshipKey, selfService, selfUserId])

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

  const resources = useMemo(() => key?.effectiveAccess ?? [], [key])
  const resourceResolutions = useMemo(
    () =>
      routingCatalog === undefined ? {} : apiKeyResourceResolutions(resources, routingCatalog),
    [resources, routingCatalog],
  )

  const quotaSource = key?.quota?.source || key?.budgetPolicySource
  const quotaSourceLabel =
    quotaSource === 'key'
      ? 'Key limit'
      : quotaSource === 'user'
        ? 'User default'
        : quotaSource === 'team'
          ? 'Team default'
          : ''
  const effectiveCanManage = Boolean(canManage || (selfService && canSelfManage))
  const model = apiKeyQuickstartModel(resources, resourceResolutions)
  const visibleResourceNames = useMemo(
    () => apiKeyVisibleResourceNames(resources, resourceResolutions),
    [resourceResolutions, resources],
  )
  // Inference is served by the Router's public listener, never by the Dashboard.
  const baseURL = routerPublicEndpoint(routerPublicUrl, '/v1')
  const snippets = useMemo(
    () => (model ? buildAPIKeyQuickstartSnippets(baseURL, model, secret) : null),
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
        className={styles.detailDialog}
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
                  ? `${key.prefix}•••••••• · ${ownerName || key.ownerId}`
                  : 'Loading key details…'}
              </p>
            </div>
          </div>
          <button type="button" className={styles.modalClose} onClick={onClose} aria-label="Close">
            <ProductIcon name="close" />
          </button>
        </header>
        <div className={styles.detailBody}>
          {loading ? <ProductLoadingState compact label="Loading API key details" /> : null}
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
                  {canReveal ? (
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
                    <dd>{ownerName || key.ownerId}</dd>
                  </div>
                  <div>
                    <dt>Budget</dt>
                    <dd>
                      {key.quota
                        ? `${assignedBudget?.name || key.quota.budgetName} · ${quotaSourceLabel}`
                        : assignedBudget
                          ? `${assignedBudget.name} · ${quotaSourceLabel}`
                          : 'No quota policy'}
                    </dd>
                  </div>
                  <div>
                    <dt>Team context</dt>
                    <dd>
                      {key.contextTeamId ? contextTeamName || key.contextTeamId : 'Personal policy'}
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
                      {routingCatalog === null ? (
                        <span className={styles.inlineActions}>
                          Models unavailable
                          <button
                            type="button"
                            onClick={() => setRoutingCatalogAttempt((attempt) => attempt + 1)}
                          >
                            Retry
                          </button>
                        </span>
                      ) : visibleResourceNames.length ? (
                        visibleResourceNames.map((name) => (
                          <code key={name}>
                            {name}
                          </code>
                        ))
                      ) : (
                        'No models assigned'
                      )}
                    </dd>
                  </div>
                  <div>
                    <dt>Access groups</dt>
                    <dd>
                      {assignedGroups.length
                        ? assignedGroups.map((group) => group.name).join(', ')
                        : key.accessGroupIds.length
                          ? key.accessGroupIds.join(', ')
                          : 'Not assigned'}
                    </dd>
                  </div>
                  <div>
                    <dt>Access sources</dt>
                    <dd>
                      {key.accessPolicySources?.length
                        ? key.accessPolicySources.join(' + ')
                        : 'None'}
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
                      disabled={!snippets}
                    >
                      {item === 'javascript' ? 'JavaScript' : item === 'python' ? 'Python' : 'cURL'}
                    </button>
                  ))}
                  <button
                    type="button"
                    className={styles.codeCopy}
                    onClick={() => (snippets ? void copy(snippets[snippet], 'snippet') : undefined)}
                    disabled={!snippets}
                  >
                    <ProductIcon name="copy" />
                    {copied === 'snippet' ? 'Copied' : 'Copy code'}
                  </button>
                </div>
                {snippets ? (
                  <pre className={styles.codeBlock}>
                    <code>{snippets[snippet]}</code>
                  </pre>
                ) : (
                  <div className={styles.codeHint} role="status">
                    {routingCatalog === undefined
                      ? 'Loading an allowed request model…'
                      : 'No request-ready model is available for this key.'}
                    {routingCatalog !== undefined ? (
                      <button
                        type="button"
                        onClick={() => setRoutingCatalogAttempt((attempt) => attempt + 1)}
                      >
                        Retry
                      </button>
                    ) : null}
                  </div>
                )}
                {snippets && !secret ? (
                  <p className={styles.codeHint}>
                    Reveal the key to place it directly in the sample. Environment variables remain
                    the safer default.
                  </p>
                ) : null}
              </section>
            </>
          ) : null}
        </div>
        {key && effectiveCanManage ? (
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
