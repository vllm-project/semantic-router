import { useState } from 'react'
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
import { useInstanceDeployment, type InstanceMode } from './useInstanceDeployment'
import styles from './DecisionModelPage.module.css'

export default function DecisionModelPage() {
  const { user } = useAuth()
  const { isReadonly, isLoading: accessLoading } = useReadonly()
  const instance = useInstanceDeployment()
  const model = useDecisionModelManagement(instance.status?.observed_mode === 'engine')
  const tasks = useDecisionTasks()
  const [modeChoice, setModeChoice] = useState<InstanceMode | null>(null)
  const [deployError, setDeployError] = useState<string | null>(null)
  const observedMode =
    instance.status?.observed_mode === 'unknown'
      ? model.status?.serving_mode
      : (instance.status?.observed_mode ?? model.status?.serving_mode)
  const mode = modeChoice ?? (observedMode === 'engine' ? 'engine' : 'router')
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
  const deploy = async () => {
    setDeployError(null)
    try {
      const next = await model.deploy()
      if (next && instance.status?.can_switch)
        await instance.deploy(mode, configuredDecisionDeployment(next))
      else if (mode !== observedMode)
        throw new Error(
          instance.status?.unavailable_reason ||
            'This instance is managed externally. Change its mode through its deployment owner.',
        )
    } catch (cause) {
      setDeployError(cause instanceof Error ? cause.message : 'Deployment failed.')
    }
  }
  return (
    <ConfigPageManagerLayout
      eyebrow="Build / System One"
      title="Decision Models"
      description="Choose the model and instance mode, understand supported tasks, and manage their bindings."
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
              disabled={model.refreshing || instance.busy}
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
              <h2 id="decision-model-status-title">{activeLabel || 'Model not reported'}</h2>
            </div>
            <span className={styles.modeBadge}>
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
              <strong>{runtimeState}</strong>
            </div>
            <div>
              <span>Deployment phase</span>
              <strong>{phase.replace(/_/g, ' ')}</strong>
            </div>
            <div>
              <span>Saved configuration</span>
              <strong>{decisionActivationLabel(model.activation)}</strong>
            </div>
          </div>
          {instance.busy && (
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
          <DecisionModelCatalog
            model={model}
            writable={writable && !instance.error}
            mode={mode}
            onModeChange={setModeChoice}
            canSwitch={Boolean(instance.status?.can_switch && !instance.error)}
            busy={instance.busy}
            onDeploy={() => void deploy()}
            tasks={tasks.data}
          />
          {model.applyResult && (
            <p className={styles.notice} role="status">
              {model.applyResult.message}
            </p>
          )}
          {(deployError || model.applyError) && (
            <p className={styles.notice} role="alert">
              {deployError || model.applyError}
            </p>
          )}
        </section>
        <DecisionTaskBindings
          data={tasks.data}
          error={tasks.error}
          writable={writable && !model.deploying && !instance.busy}
          save={(binding, deployment) =>
            model.updateConfig((current) => withDecisionTaskBinding(current, binding, deployment))
          }
        />
      </div>
    </ConfigPageManagerLayout>
  )
}
