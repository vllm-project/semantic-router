import type { ModelRuntimeDeployment } from './decisionModelManagement'
import styles from './DecisionReplicaPanel.module.css'

export default function DecisionReplicaStatus({
  deployment,
}: {
  deployment?: ModelRuntimeDeployment
}) {
  if (!deployment?.replicas?.length)
    return <p className={styles.hint}>Replica observations are not available yet.</p>
  return (
    <div className={styles.workers}>
      {deployment.replicas.map((replica) => (
        <div className={styles.worker} key={replica.id}>
          <div className={styles.workerHeading}>
            <strong>
              {replica.device || (replica.managed ? 'Managed worker' : 'Attached worker')}
            </strong>
            <span className={styles.state} data-ready={replica.ready}>
              {replica.state}
            </span>
          </div>
          <dl>
            <div>
              <dt>In flight</dt>
              <dd>{replica.inflight}</dd>
            </div>
            <div>
              <dt>Restarts</dt>
              <dd>{replica.restarts}</dd>
            </div>
          </dl>
          <small title={replica.id}>{replica.id}</small>
          {replica.reason && <p className={styles.hint}>{replica.reason}</p>}
        </div>
      ))}
    </div>
  )
}
