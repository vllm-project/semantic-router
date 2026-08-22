import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
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
  const [tab, setTab] = useState<'request' | 'response' | 'metadata'>('request')
  const [copied, setCopied] = useState(false)
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
  const payload =
    tab === 'request' ? metadata.request : tab === 'response' ? metadata.response : metadata
  const payloadText =
    payload === undefined
      ? ''
      : typeof payload === 'string'
        ? payload
        : JSON.stringify(payload, null, 2)
  const identity = log?.teamId
    ? teams.find((team) => team.id === log.teamId)?.name || log.teamId
    : users.find((user) => user.id === log?.userId)?.name || log?.userId || 'Unassigned'
  const keyName = keys.find((key) => key.id === log?.keyId)?.name || log?.keyId || 'Unknown key'

  return (
    <div
      className={styles.detailBackdrop}
      onMouseDown={(event) => {
        if (event.target === event.currentTarget) onClose()
      }}
    >
      <aside
        ref={dialogRef}
        className={styles.detailDrawer}
        role="dialog"
        aria-modal="true"
        aria-labelledby="log-detail-title"
        tabIndex={-1}
      >
        <header className={styles.detailHeader}>
          <div>
            <span>Request Log</span>
            <h2 id="log-detail-title">{log?.model || 'Request details'}</h2>
            <p>{log?.requestId || 'Loading request…'}</p>
          </div>
          <button type="button" className={styles.modalClose} onClick={onClose} aria-label="Close">
            ×
          </button>
        </header>
        <div className={styles.detailBody}>
          {loading ? <div className={styles.detailLoading}>Loading request details…</div> : null}
          {error ? (
            <div className={styles.modalError} role="alert">
              <span>!</span>
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
                <button
                  type="button"
                  onClick={() => {
                    void copyText(log.requestId).then((ok) => {
                      setCopied(ok)
                      if (ok) window.setTimeout(() => setCopied(false), 1600)
                    })
                  }}
                >
                  {copied ? 'Copied' : 'Copy request ID'}
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
                    <dt>Model</dt>
                    <dd>{log.model}</dd>
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
                  <span>Payload</span>
                  <h3>Request & response</h3>
                </div>
                <div className={styles.codeTabs}>
                  {(['request', 'response', 'metadata'] as const).map((item) => (
                    <button
                      type="button"
                      key={item}
                      className={tab === item ? styles.codeTabActive : ''}
                      onClick={() => setTab(item)}
                    >
                      {item}
                    </button>
                  ))}
                  {payloadText ? (
                    <button
                      type="button"
                      className={styles.codeCopy}
                      onClick={() => void copyText(payloadText)}
                    >
                      Copy
                    </button>
                  ) : null}
                </div>
                {metadata.payloadRedacted ? (
                  <div className={styles.payloadNotice}>
                    Request and response bodies are visible to workspace builders and
                    administrators.
                  </div>
                ) : payloadText ? (
                  <pre className={`${styles.codeBlock} ${styles.logPayload}`}>
                    <code>{payloadText}</code>
                  </pre>
                ) : (
                  <div className={styles.payloadNotice}>
                    No payload was recorded for this request.
                  </div>
                )}
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
              Open in Insights
            </Link>
            <button type="button" className={styles.primaryButton} onClick={onClose}>
              Done
            </button>
          </footer>
        ) : null}
      </aside>
    </div>
  )
}
