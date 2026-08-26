import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import ProductIcon from '../components/ProductIcon'
import ProductLoadingState from '../components/ProductLoadingState'
import useAccessibleDialog from '../hooks/useAccessibleDialog'
import { copyText } from '../utils/clipboard'
import {
  inferenceAccessApi,
  type AccessAPIKey,
  type AccessTeam,
  type AccessUsageEvent,
  type AccessUser,
} from '../utils/inferenceAccessApi'
import { formatDate, formatNumber } from './AccessControlDetailSupport'
import styles from './AccessControlPage.module.css'

interface LogDetailProps {
  logId: string
  users: AccessUser[]
  teams: AccessTeam[]
  keys: AccessAPIKey[]
  selfService?: boolean
  onClose: () => void
}

export function RequestLogDetail({
  logId,
  users,
  teams,
  keys,
  selfService = false,
  onClose,
}: LogDetailProps) {
  const [log, setLog] = useState<AccessUsageEvent | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [copied, setCopied] = useState<'request' | 'payload' | ''>('')
  const dialogRef = useAccessibleDialog<HTMLDivElement>({ isOpen: true, onClose })

  useEffect(() => {
    setLoading(true)
    setError('')
    void (
      selfService ? inferenceAccessApi.selfRequestLog(logId) : inferenceAccessApi.requestLog(logId)
    )
      .then(setLog)
      .catch((nextError) =>
        setError(nextError instanceof Error ? nextError.message : 'Could not load request'),
      )
      .finally(() => setLoading(false))
  }, [logId, selfService])

  const metadata = log?.metadata || {}
  const metadataText = Object.keys(metadata).length > 0 ? JSON.stringify(metadata, null, 2) : ''
  const route = log?.routing || {}
  const routeValue = (key: string) => {
    const value = route[key]
    return typeof value === 'string' && value.trim() ? value : undefined
  }
  const modelNames = log?.models.map((model) => model.name || model.id).filter(Boolean) ?? []
  const selectedModels = log ? modelNames.join(', ') || log.model || 'Unassigned' : ''
  const identity = log?.teamId
    ? teams.find((team) => team.id === log.teamId)?.name || log.teamId
    : users.find((user) => user.id === log?.userId)?.name || log?.userId || 'Unassigned'
  const keyName = keys.find((key) => key.id === log?.keyId)?.name || log?.keyId || 'Unknown key'
  const copyValue = async (value: string, target: 'request' | 'payload') => {
    if (!(await copyText(value))) return
    setCopied(target)
    window.setTimeout(() => setCopied(''), 1600)
  }

  return (
    <div
      className={styles.detailBackdrop}
      onMouseDown={(event) => {
        if (event.target === event.currentTarget) onClose()
      }}
    >
      <aside
        ref={dialogRef}
        className={styles.detailDialog}
        role="dialog"
        aria-modal="true"
        aria-labelledby="log-detail-title"
        aria-busy={loading}
        tabIndex={-1}
      >
        <header className={styles.detailHeader}>
          <div className={styles.detailHeaderIdentity}>
            <div className={styles.modalLogo} aria-hidden="true">
              <img src="/vllm.png" alt="" />
            </div>
            <div>
              <span>Request log</span>
              <h2 id="log-detail-title">{selectedModels || 'Request details'}</h2>
              <p>{log?.requestId || 'Loading request…'}</p>
            </div>
          </div>
          <button type="button" className={styles.modalClose} onClick={onClose} aria-label="Close">
            <ProductIcon name="close" />
          </button>
        </header>
        <div className={styles.detailBody}>
          {loading ? <ProductLoadingState compact label="Loading request details" /> : null}
          {error ? (
            <div className={styles.modalError} role="alert">
              <ProductIcon name="alert" aria-hidden="true" />
              <div>
                <strong>Couldn’t load request</strong>
                <p>{error}</p>
              </div>
            </div>
          ) : null}
          {log ? (
            <>
              <div className={styles.logHero}>
                <div>
                  <span
                    className={
                      log.statusCode < 400 ? styles.detailStatusLive : styles.detailStatusOff
                    }
                  >
                    {log.statusCode}
                  </span>
                  <strong>{log.statusCode < 400 ? 'Completed' : 'Failed'}</strong>
                  <small>{formatDate(log.createdAt)}</small>
                </div>
                <button type="button" onClick={() => void copyValue(log.requestId, 'request')}>
                  <ProductIcon name={copied === 'request' ? 'check' : 'copy'} />
                  {copied === 'request' ? 'Copied' : 'Copy request ID'}
                </button>
              </div>
              <div className={styles.detailMetrics}>
                <article>
                  <span>Latency</span>
                  <strong>{formatNumber(log.latencyMs)} ms</strong>
                </article>
                <article>
                  <span>First token</span>
                  <strong>{log.ttftMs ? `${formatNumber(log.ttftMs)} ms` : '—'}</strong>
                </article>
                <article>
                  <span>Input tokens</span>
                  <strong>{formatNumber(log.promptTokens)}</strong>
                </article>
                <article>
                  <span>Output tokens</span>
                  <strong>{formatNumber(log.completionTokens)}</strong>
                </article>
              </div>

              <section className={styles.detailSection}>
                <div className={styles.detailSectionHeading}>
                  <span>Route</span>
                  <h3>Request context</h3>
                </div>
                <dl className={styles.detailGrid}>
                  <div>
                    <dt>Identity</dt>
                    <dd>{identity}</dd>
                  </div>
                  <div>
                    <dt>API key</dt>
                    <dd>{keyName}</dd>
                  </div>
                  <div>
                    <dt>Entrypoint</dt>
                    <dd>{routeValue('entrypointName') || log.entrypointId || '—'}</dd>
                  </div>
                  <div>
                    <dt>Recipe</dt>
                    <dd>{routeValue('recipeName') || log.recipeId || '—'}</dd>
                  </div>
                  <div>
                    <dt>Decision</dt>
                    <dd>{log.decisionName || log.decisionId || '—'}</dd>
                  </div>
                  <div>
                    <dt>Selected model</dt>
                    <dd>{selectedModels}</dd>
                  </div>
                  <div>
                    <dt>Total tokens</dt>
                    <dd>{formatNumber(log.totalTokens)}</dd>
                  </div>
                  {log.errorCode ? (
                    <div className={styles.detailGridWide}>
                      <dt>Error</dt>
                      <dd>{log.errorCode}</dd>
                    </div>
                  ) : null}
                </dl>
              </section>

              <section className={styles.detailSection}>
                <div className={styles.detailSectionHeading}>
                  <span>Execution</span>
                  <h3>Model dispatches</h3>
                </div>
                {log.dispatches?.length ? (
                  <div className={styles.logDispatchList}>
                    {log.dispatches.map((dispatch) => (
                      <article key={dispatch.dispatchId} className={styles.logDispatch}>
                        <div className={styles.logDispatchHeader}>
                          <div>
                            <strong>
                              {log.models.find((model) => model.id === dispatch.modelId)?.name ||
                                dispatch.providerModelId ||
                                dispatch.modelId ||
                                dispatch.dispatchType}
                            </strong>
                            <span>
                              {dispatch.providerId || 'Router'} · {dispatch.dispatchType}
                            </span>
                          </div>
                          <span>{dispatch.usageState.replace(/_/g, ' ')}</span>
                        </div>
                        <dl>
                          <div>
                            <dt>Input</dt>
                            <dd>{formatNumber(Number(dispatch.inputTokens))}</dd>
                          </div>
                          <div>
                            <dt>Output</dt>
                            <dd>{formatNumber(Number(dispatch.outputTokens))}</dd>
                          </div>
                          <div>
                            <dt>Attempts</dt>
                            <dd>{dispatch.attempts.length}</dd>
                          </div>
                          <div>
                            <dt>Backend</dt>
                            <dd>{dispatch.backendId || '—'}</dd>
                          </div>
                        </dl>
                      </article>
                    ))}
                  </div>
                ) : (
                  <div className={styles.payloadNotice}>
                    Dispatch details are restricted for this account.
                  </div>
                )}
              </section>

              {log.quotaReceipts?.length ? (
                <section className={styles.detailSection}>
                  <div className={styles.detailSectionHeading}>
                    <span>Quota</span>
                    <h3>Accounted usage</h3>
                  </div>
                  <div className={styles.logReceiptList}>
                    {log.quotaReceipts.map((receipt, index) => (
                      <div key={`${String(receipt.ruleId ?? '')}-${index}`}>
                        <strong>{String(receipt.metric ?? 'Usage')}</strong>
                        <span>{String(receipt.amount ?? '—')}</span>
                      </div>
                    ))}
                  </div>
                </section>
              ) : null}

              <section className={styles.detailSection}>
                <div className={styles.detailSectionHeading}>
                  <span>Evidence</span>
                  <h3>Request metadata</h3>
                </div>
                {metadataText ? (
                  <>
                    <div className={styles.codeTabs}>
                      <span>Metadata</span>
                      <button
                        type="button"
                        className={styles.codeCopy}
                        onClick={() => void copyValue(metadataText, 'payload')}
                      >
                        <ProductIcon name={copied === 'payload' ? 'check' : 'copy'} />
                        {copied === 'payload' ? 'Copied' : 'Copy'}
                      </button>
                    </div>
                    <pre className={`${styles.codeBlock} ${styles.logPayload}`}>
                      <code>{metadataText}</code>
                    </pre>
                  </>
                ) : null}
                <div className={styles.payloadNotice}>
                  Request and response bodies are not retained in the usage ledger.
                </div>
              </section>
            </>
          ) : null}
        </div>
        {log ? (
          <footer className={styles.detailFooter}>
            <Link
              className={styles.secondaryLink}
              to={`/insights?search=${encodeURIComponent(log.requestId)}`}
            >
              <ProductIcon name="insight" aria-hidden="true" />
              Open in Insights
            </Link>
            <button type="button" className={styles.primaryButton} onClick={onClose}>
              <ProductIcon name="check" /> Done
            </button>
          </footer>
        ) : null}
      </aside>
    </div>
  )
}
