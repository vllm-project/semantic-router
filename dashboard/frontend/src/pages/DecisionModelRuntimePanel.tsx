import { useAuth } from '../contexts/AuthContext'
import { canAccessDashboardPath } from '../utils/accessControl'
import type { ModelRuntimeInventory } from './decisionModelManagement'
import { DECISION_MODEL_METRICS, formatDecisionModelMetric } from './decisionModelMetrics'
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
  const canReadMetrics = canAccessDashboardPath(user, '/monitoring')
  const deployments = inventory?.deployments ?? []
  const metrics = useDecisionModelMetrics(
    deployments.map(({ name }) => name),
    refreshedAt,
    canReadMetrics && !engineOnly,
  )

  return (
    <section className={styles.panel} aria-labelledby="decision-runtime-title">
      <h2 id="decision-runtime-title">Model runtime deployments</h2>
      <p className={styles.muted}>
        Decision models and specialist dependencies reported by the router. Statistics cover the
        last 5 minutes across routers scraped by Prometheus. Runtime calls can share a question
        batch; they do not count backend LLM chat requests.
      </p>
      {!canReadMetrics ? (
        <p className={styles.muted}>Viewing model statistics requires observability read access.</p>
      ) : engineOnly ? (
        <p className={styles.muted}>
          Router call statistics are unavailable in standalone engine mode.
        </p>
      ) : metrics.unavailable.length > 0 ? (
        <p className={styles.muted}>
          Some model statistics are unavailable. Check the Prometheus connection and router scrape.
        </p>
      ) : null}
      <p className={styles.muted}>
        {metrics.loading ? 'Updating statistics… ' : ''}
        Unreported values have no usable samples; latency and percentages need traffic. Token counts
        and GPU memory are not included in these router metrics.
      </p>
      {!deployments.length && <p>No model runtime deployments reported.</p>}
      <div className={styles.deployments}>
        {deployments.map((deployment) => (
          <article className={styles.deployment} key={deployment.name} aria-label={deployment.name}>
            <div className={styles.deploymentTitle}>
              <h3>{deployment.name}</h3>
              <span>
                {deployment.ready && deployment.state === 'ready'
                  ? 'Ready'
                  : deployment.state || 'Not ready'}
              </span>
            </div>
            {deployment.reason && <p className={styles.notice}>{deployment.reason}</p>}
            <dl className={styles.statistics} aria-label="Model statistics">
              {DECISION_MODEL_METRICS.map((metric) => (
                <div key={metric.key}>
                  <dt title={metric.help}>{metric.label}</dt>
                  <dd>
                    {formatDecisionModelMetric(
                      metrics.values[metric.key]?.[deployment.name],
                      metric.unit,
                    )}
                  </dd>
                </div>
              ))}
            </dl>
            <dl className={styles.facts}>
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
                <dt>Repository</dt>
                <dd>{shown(deployment.repo)}</dd>
              </div>
            </dl>
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
                <div>
                  <dt>Ownership</dt>
                  <dd>{deployment.managed ? 'Managed by router' : 'External deployment'}</dd>
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
        ))}
      </div>
    </section>
  )
}
