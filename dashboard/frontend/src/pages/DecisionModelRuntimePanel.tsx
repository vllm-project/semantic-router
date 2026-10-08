import { useState } from 'react'
import { useAuth } from '../contexts/AuthContext'
import { canAccessDashboardPath } from '../utils/accessControl'
import type { ModelRuntimeInventory } from './decisionModelManagement'
import {
  DECISION_MODEL_TIME_WINDOWS,
  decisionModelChartPoints,
  type DecisionModelTimeWindow,
} from './decisionModelMetrics'
import DecisionModelMonitoringCharts, {
  DecisionModelMetricCards,
} from './DecisionModelMonitoringCharts'
import { useDecisionModelMetrics } from './useDecisionModelMetrics'
import styles from './DecisionModelPage.module.css'

const shown = (value?: string) => value?.trim() || 'Not reported'

export default function DecisionModelRuntimePanel({
  inventory,
  refreshedAt,
  engineOnly,
}: {
  inventory: ModelRuntimeInventory | null
  refreshedAt: Date | null
  engineOnly: boolean
}) {
  const { user } = useAuth()
  const [timeWindow, setTimeWindow] = useState<DecisionModelTimeWindow>(3_600)
  const [selectedDeployment, setSelectedDeployment] = useState('')
  const canReadMetrics = canAccessDashboardPath(user, '/monitoring')
  const deployments = inventory?.deployments ?? []
  const deployment = deployments.find(({ name }) => name === selectedDeployment) ?? deployments[0]
  const metrics = useDecisionModelMetrics(
    deployment ? [deployment.name] : [],
    refreshedAt,
    canReadMetrics && !engineOnly,
    timeWindow,
  )
  const points = deployment ? decisionModelChartPoints(metrics.series, deployment.name) : []
  const ready = deployment?.ready && deployment.state === 'ready'

  return (
    <section
      className={`${styles.panel} ${styles.monitoringPanel}`}
      aria-labelledby="decision-runtime-title"
    >
      <div className={styles.monitoringHeader}>
        <div>
          <span className={styles.eyebrow}>Live telemetry</span>
          <h2 id="decision-runtime-title">Model runtime deployments</h2>
          <p className={styles.muted}>
            Follow traffic, latency, and cache behavior for each runtime.
          </p>
        </div>
        <div className={styles.timeWindows} role="group" aria-label="Monitoring time range">
          {DECISION_MODEL_TIME_WINDOWS.map((window) => (
            <button
              key={window.seconds}
              type="button"
              aria-pressed={timeWindow === window.seconds}
              onClick={() => setTimeWindow(window.seconds)}
            >
              {window.label}
            </button>
          ))}
        </div>
      </div>
      {!canReadMetrics ? (
        <p className={styles.notice}>
          Viewing model statistics requires observability read access.
        </p>
      ) : engineOnly ? (
        <p className={styles.notice}>
          Router call statistics are unavailable in standalone engine mode.
        </p>
      ) : metrics.unavailable.length > 0 ? (
        <p className={styles.notice}>
          Some model statistics are unavailable. Check the Prometheus connection and router scrape.
          <span className={styles.unavailableMetrics}>
            Unavailable: {metrics.unavailable.join(', ')}.
          </span>
        </p>
      ) : null}
      {!deployment ? (
        <div className={styles.runtimeEmpty}>
          <strong>No model runtime deployments reported.</strong>
          <p>Deploy a decision model to inspect its runtime and collected observations.</p>
        </div>
      ) : (
        <article className={styles.deployment} key={deployment.name} aria-label={deployment.name}>
          <div className={styles.deploymentTitle}>
            <div className={styles.deploymentIdentity}>
              <div className={styles.runtimeIcon} aria-hidden="true">
                <svg
                  width="22"
                  height="22"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="1.5"
                >
                  <rect x="5" y="5" width="14" height="14" rx="3" />
                  <rect x="9" y="9" width="6" height="6" rx="1" />
                  <path d="M9 1v4m6-4v4M9 19v4m6-4v4M1 9h4m-4 6h4m14-6h4m-4 6h4" />
                </svg>
              </div>
              <div>
                {deployments.length > 1 ? (
                  <select
                    aria-label="Runtime deployment"
                    className={styles.deploymentSelect}
                    value={deployment.name}
                    onChange={(event) => setSelectedDeployment(event.target.value)}
                  >
                    {deployments.map((candidate) => (
                      <option value={candidate.name} key={candidate.name}>
                        {candidate.name}
                      </option>
                    ))}
                  </select>
                ) : (
                  <h3>{deployment.name}</h3>
                )}
                <p className={styles.repository}>{shown(deployment.repo)}</p>
              </div>
            </div>
            <span className={`${styles.runtimeState} ${ready ? styles.ready : styles.pending}`}>
              <i />
              {ready ? 'Ready' : deployment.state || 'Not ready'}
            </span>
          </div>
          {deployment.reason && <p className={styles.notice}>{deployment.reason}</p>}
          <dl className={styles.runtimeMeta}>
            <div>
              <dt>Device</dt>
              <dd>{shown(deployment.device)}</dd>
            </div>
            <div>
              <dt>Engine backend</dt>
              <dd>{shown(deployment.engine)}</dd>
            </div>
            <div>
              <dt>Restarts</dt>
              <dd>{deployment.restarts ?? 'Not reported'}</dd>
            </div>
            <div>
              <dt>Ownership</dt>
              <dd>{deployment.managed ? 'Managed by router' : 'External deployment'}</dd>
            </div>
          </dl>
          {canReadMetrics && !engineOnly && (
            <>
              <div className={styles.metricsCaption}>
                <span>
                  Latest observation <span className={styles.captionSeparator}>/</span> 5-minute
                  rolling rates
                </span>
                <span>
                  {metrics.loading
                    ? 'Updating statistics…'
                    : metrics.updatedAt
                      ? `Observed ${metrics.updatedAt.toLocaleTimeString()}`
                      : 'Waiting for samples'}
                </span>
              </div>
              <DecisionModelMetricCards points={points} />
              <DecisionModelMonitoringCharts points={points} loading={metrics.loading} />
              <p className={styles.metricsFootnote}>
                Aggregated across routers scraped by Prometheus. Runtime calls may share a question
                batch; they do not count backend LLM chat requests. Gaps mean no usable samples, not
                zero. Token counts and GPU memory are not reported by these metrics.
              </p>
            </>
          )}
          <details className={styles.hashes}>
            <summary>Deployment details</summary>
            <dl className={styles.facts}>
              <div>
                <dt>Revision</dt>
                <dd>{shown(deployment.revision)}</dd>
              </div>
              <div>
                <dt>Profile</dt>
                <dd>{shown(deployment.profile)}</dd>
              </div>
              <div>
                <dt>Family</dt>
                <dd>{shown(deployment.family)}</dd>
              </div>
              <div>
                <dt>Process</dt>
                <dd>{shown(deployment.process)}</dd>
              </div>
              <div>
                <dt>Served model</dt>
                <dd>{shown(deployment.served_name)}</dd>
              </div>
            </dl>
            {deployment.surfaces?.length ? (
              <p>Supported APIs: {deployment.surfaces.join(', ')}</p>
            ) : null}
            {deployment.heads?.length ? (
              <p>Model heads: {deployment.heads.map((head) => head.name).join(', ')}</p>
            ) : null}
          </details>
        </article>
      )}
    </section>
  )
}
