import { Link } from 'react-router-dom'
import { useAuth } from '../contexts/AuthContext'
import { useReadonly } from '../contexts/ReadonlyContext'
import { canWriteConfig } from '../utils/accessControl'
import ConfigPageManagerLayout from './ConfigPageManagerLayout'
import { decisionActivationLabel, decisionModelRuntimeState } from './decisionModelManagement'
import { configuredDecisionDeployment, DECISION_MODEL_OPTIONS } from './decisionModelSupport'
import DecisionModelCatalog from './DecisionModelCatalog'
import DecisionTaskBindings, { withDecisionTaskBinding } from './DecisionTaskBindings'
import { useDecisionModelManagement } from './useDecisionModelManagement'
import { useDecisionTasks } from './useDecisionTasks'
import { useInstanceDeployment } from './useInstanceDeployment'
import DecisionReplicaPanel from './DecisionReplicaPanel'
import DecisionObservationStatus from './DecisionObservationStatus'
import styles from './DecisionModelPage.module.css'

export default function DecisionModelPage() {
  const { user } = useAuth()
  const { isReadonly, isLoading: accessLoading } = useReadonly()
  const instance = useInstanceDeployment()
  const instanceBusy = !instance.stale && instance.busy
  const model = useDecisionModelManagement()
  const tasks = useDecisionTasks()
  const observedMode =
    instance.status?.observed_mode === 'unknown'
      ? model.status?.serving_mode
      : (instance.status?.observed_mode ?? model.status?.serving_mode)
  const writable = !isReadonly && !accessLoading && canWriteConfig(user)
  const savedLabel =
    DECISION_MODEL_OPTIONS.find((option) => option.name === model.savedModel)?.label ??
    model.savedModel
  const activeDeployment =
    instance.status?.active_deployment ||
    (model.config ? configuredDecisionDeployment(model.config) : '')
  const observed = model.inventory?.deployments.find((entry) => entry.name === activeDeployment)
  const runtimeState = decisionModelRuntimeState(activeDeployment, model.inventory)
  const activeLabel =
    observed?.repo?.split('/').slice(-1)[0] || instance.status?.model || observed?.served_name
  const operation = instance.status?.operation
  const phase = operation?.phase ?? model.status?.router_runtime?.phase ?? 'Not reported'
  return (
    <ConfigPageManagerLayout
      eyebrow="Build / System One"
      title="Decision Models"
      description="Manage decision models, runtime capacity and task bindings. Serving mode follows the instance startup configuration."
    >
      <div className={styles.page}>
        <div className={styles.toolbar}>
          <span>
            {model.updatedAt
              ? `Observed ${model.updatedAt.toLocaleTimeString()}`
              : 'Loading deployment observations…'}
          </span>
          <div className={styles.toolbarActions}>
            <Link className={styles.testLink} to="/decision-model/playground">
              Decision Playground
            </Link>
            <Link className={styles.testLink} to="/decision-model/monitoring">
              Decision Monitoring
            </Link>
            <button
              type="button"
              disabled={model.refreshing || instanceBusy}
              onClick={() => {
                void model.refresh()
                void instance.refresh()
                tasks.refresh()
              }}
            >
              Refresh
            </button>
          </div>
        </div>
        {model.errors.length > 0 && (
          <details className={styles.notice}>
            <summary>Some deployment observations are unavailable</summary>
            {model.errors.map((error) => (
              <p key={error}>{error}</p>
            ))}
          </details>
        )}
        <section
          className={`${styles.panel} ${styles.statusPanel}`}
          aria-labelledby="decision-model-status-title"
        >
          <div className={styles.deploymentHeading}>
            <div>
              <span className={styles.statusEyebrow}>Current deployment</span>
              <h2 id="decision-model-status-title">
                {activeLabel ||
                  (instance.loading || model.observationState.inventory.loading
                    ? 'Loading active model…'
                    : 'Model not reported')}
              </h2>
            </div>
            <span className={styles.modeBadge}>
              {instance.stale ? 'Last observed: ' : ''}
              {observedMode === 'router'
                ? 'Router mode'
                : observedMode === 'engine'
                  ? 'Engine mode'
                  : 'Mode unavailable'}
            </span>
          </div>
          <div className={styles.deploymentFacts}>
            <div>
              <span>Model readiness</span>
              <strong>
                {model.observationState.inventory.loading
                  ? 'Checking runtime…'
                  : `${model.observationState.inventory.stale ? 'Last observed: ' : ''}${runtimeState}`}
              </strong>
              <DecisionObservationStatus
                label="Runtime"
                observation={model.observationState.inventory}
              />
            </div>
            <div>
              <span>Deployment phase</span>
              <strong>
                {instance.loading
                  ? 'Checking instance…'
                  : `${instance.stale ? 'Last observed: ' : ''}${phase.replace(/_/g, ' ')}`}
              </strong>
              <DecisionObservationStatus label="Instance" observation={instance} />
            </div>
            <div>
              <span>Saved configuration</span>
              <strong>
                {model.observationState.activation.loading
                  ? 'Checking configuration…'
                  : `${model.observationState.activation.stale ? 'Last observed: ' : ''}${decisionActivationLabel(model.activation)}`}
              </strong>
              <DecisionObservationStatus
                label="Activation"
                observation={model.observationState.activation}
              />
            </div>
          </div>
          {instanceBusy && (
            <div className={styles.deploymentProgress} role="status">
              <span />
              Deployment in progress. This page remains available while the runtime changes.
            </div>
          )}
          {operation?.error && (
            <p className={styles.notice} role="alert">
              {operation.error}
              {operation.rolled_back ? ' The previous deployment was restored.' : ''}
            </p>
          )}
          <details className={styles.hashes}>
            <summary>Deployment diagnostics</summary>
            <p>Saved model: {savedLabel || 'Not reported'}</p>
            <p>
              Default deployment:{' '}
              {model.config
                ? configuredDecisionDeployment(model.config) || 'Not reported'
                : 'Loading'}
            </p>
            <p>Managed by: {instance.status?.ownership || 'Not reported'}</p>
            {instance.status?.unavailable_reason && <p>{instance.status.unavailable_reason}</p>}
            {instance.error && <p>{instance.error}</p>}
            {model.status?.router_runtime?.message && <p>{model.status.router_runtime.message}</p>}
            {model.activation?.activation?.reasons?.map((reason, index) => (
              <p key={index}>{reason.message || reason.code}</p>
            ))}
            <p>
              Generated configuration:{' '}
              <code>{model.activation?.generated_runtime_hash || 'Not reported'}</code>
            </p>
            <p>
              Active configuration:{' '}
              <code>{model.activation?.active_runtime_hash || 'Not reported'}</code>
            </p>
          </details>
        </section>
        <section className={styles.panel} aria-labelledby="decision-model-choose-title">
          <DecisionObservationStatus
            label="Model configuration"
            observation={model.observationState.global}
          />
          <DecisionModelCatalog
            model={model}
            writable={writable}
            busy={instanceBusy}
            onDeploy={() => void model.deploy().catch(() => {})}
            tasks={tasks.data}
          />
          {model.applyResult && (
            <p className={styles.notice} role="status">
              {model.applyResult.message}
            </p>
          )}
          {model.applyError && (
            <p className={styles.notice} role="alert">
              {model.applyError}
            </p>
          )}
        </section>
        <DecisionReplicaPanel model={model} writable={writable && !instanceBusy} />
        <DecisionTaskBindings
          data={tasks.data}
          error={tasks.error}
          observation={tasks}
          writable={writable && !model.deploying && !instanceBusy && !tasks.stale}
          save={(binding, deployment) =>
            model.updateConfig((current) => withDecisionTaskBinding(current, binding, deployment))
          }
        />
      </div>
    </ConfigPageManagerLayout>
  )
}
