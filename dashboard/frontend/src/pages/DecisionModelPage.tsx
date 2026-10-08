import { Link } from 'react-router-dom'
import ProductLoadingState from '../components/ProductLoadingState'
import { useAuth } from '../contexts/AuthContext'
import { useReadonly } from '../contexts/ReadonlyContext'
import { canWriteConfig } from '../utils/accessControl'
import { getRouterModelStateLabel } from '../utils/routerRuntime'
import ConfigPageManagerLayout from './ConfigPageManagerLayout'
import { buildIntelligenceRoutingScopes } from './dashboardRouterIntelligenceSupport'
import { decisionActivationLabel, decisionModelRuntimeState } from './decisionModelManagement'
import { DECISION_MODEL_OPTIONS } from './decisionModelSupport'
import { useDecisionModelManagement } from './useDecisionModelManagement'
import styles from './DecisionModelPage.module.css'

const shown = (value?: string) => value?.trim() || 'Not reported'

export default function DecisionModelPage() {
  const { user } = useAuth()
  const { isReadonly, isLoading: accessLoading } = useReadonly()
  const model = useDecisionModelManagement()
  const engineOnly = model.status?.serving_mode === 'engine'
  const writable = !engineOnly && !isReadonly && !accessLoading && canWriteConfig(user)
  const scopes = model.config ? buildIntelligenceRoutingScopes(model.config) : []
  const consumers = model.status?.models?.models ?? []
  const bindings = Object.entries(model.global?.model_catalog?.system ?? {}).filter(
    ([name, value]) => name !== 'decision_model' && typeof value === 'string' && value.trim(),
  )
  const runtimeState = model.savedModel
    ? decisionModelRuntimeState(model.savedModel, model.inventory, model.status?.models)
    : 'Not reported'

  if (model.loading) return <ProductLoadingState label="Loading decision model management" />

  return (
    <ConfigPageManagerLayout
      eyebrow="System"
      title="Decision Model"
      description="Choose the model that powers router intelligence, deploy it, and inspect its live runtime."
    >
      <div className={styles.page}>
        <div className={styles.toolbar}>
          <span>
            {model.updatedAt
              ? `Last checked ${model.updatedAt.toLocaleTimeString()} · refreshes every 10 seconds`
              : 'Waiting for observations'}
          </span>
          <button
            type="button"
            onClick={() => void model.refresh()}
            disabled={model.refreshing || model.deploying}
          >
            {model.refreshing ? 'Refreshing…' : 'Refresh'}
          </button>
        </div>
        {model.errors.length > 0 && (
          <div className={styles.notice} role="alert">
            {model.errors.map((error) => (
              <p key={error}>{error}</p>
            ))}
          </div>
        )}

        <section className={styles.panel} aria-labelledby="decision-model-status-title">
          <h2 id="decision-model-status-title">Deployment status</h2>
          <div className={styles.summary}>
            <div>
              <span>Saved model</span>
              <strong>{model.savedModel ?? 'Unavailable'}</strong>
            </div>
            <div>
              <span>Observed model runtime</span>
              <strong>{runtimeState}</strong>
            </div>
            <div>
              <span>Serving mode</span>
              <strong>
                {model.status?.serving_mode === 'router'
                  ? 'Router'
                  : model.status?.serving_mode === 'engine'
                    ? 'Engine'
                    : 'Not reported'}
              </strong>
            </div>
            <div>
              <span>Configuration activation</span>
              <strong>{decisionActivationLabel(model.activation)}</strong>
            </div>
          </div>
          {model.status?.router_runtime && (
            <p className={styles.muted}>
              Startup: {model.status.router_runtime.phase}. {model.status.router_runtime.message}
            </p>
          )}
          {model.activation?.activation?.reasons?.map((reason, index) => (
            <p className={styles.notice} key={`${reason.code}-${index}`}>
              {reason.code}
              {reason.path ? ` · ${reason.path}` : ''}:{' '}
              {reason.message || 'No further detail reported.'}
            </p>
          ))}
          {model.activation?.activation?.error && (
            <p className={styles.notice}>{model.activation.activation.error}</p>
          )}
          <details className={styles.hashes}>
            <summary>Configuration versions</summary>
            <p>
              Generated: <code>{shown(model.activation?.generated_runtime_hash)}</code>
            </p>
            <p>
              Active: <code>{shown(model.activation?.active_runtime_hash)}</code>
            </p>
          </details>
        </section>

        <section className={styles.panel} aria-labelledby="decision-model-choose-title">
          <h2 id="decision-model-choose-title">Choose a decision model</h2>
          <p className={styles.muted}>
            Built-in signals, custom questions, and decision selectors use this model unless they
            name another deployment or binding.
          </p>
          {engineOnly && (
            <p className={styles.notice}>
              This deployment is a standalone engine. Router decision-model settings do not manage
              the standalone engine; model deployment controls are unavailable here.
            </p>
          )}
          <fieldset
            className={styles.options}
            disabled={!writable || model.deploying || !model.global}
          >
            <legend className={styles.srOnly}>Decision model selection</legend>
            {DECISION_MODEL_OPTIONS.map((option) => (
              <label
                key={option.name}
                className={`${styles.option} ${model.selectedModel === option.name ? styles.selected : ''}`}
              >
                <input
                  type="radio"
                  name="decision-model"
                  value={option.name}
                  checked={model.selectedModel === option.name}
                  onChange={() => model.selectModel(option.name)}
                />
                <span>
                  <strong>{option.label}</strong>
                  <span className={styles.hardware}>{option.hardware}</span>
                  <span>{option.summary}</span>
                  {model.savedModel === option.name && <em>Saved configuration</em>}
                  {model.selectedModel === option.name && model.savedModel !== option.name && (
                    <em>Selected · not saved</em>
                  )}
                </span>
              </label>
            ))}
          </fieldset>
          <div className={styles.deployBar}>
            <p className={styles.muted}>
              {engineOnly
                ? 'Connect this Dashboard to a router to manage its decision model.'
                : writable
                  ? 'Saves the selected model and requests runtime activation. Existing signal overrides stay unchanged.'
                  : 'This session can inspect decision models. Configuration write access is required to deploy.'}
            </p>
            <button
              className={styles.primary}
              type="button"
              disabled={!writable || model.deploying || !model.global || !model.selectedModel}
              onClick={() => void model.deploy()}
            >
              {model.deploying ? 'Deploying…' : 'Deploy selected model'}
            </button>
          </div>
          {model.applyResult && (
            <div className={styles.notice} role="status">
              <strong>
                Latest deployment request:{' '}
                {model.applyResult.status === 'success'
                  ? 'Configuration applied'
                  : model.applyResult.status === 'restart_required'
                    ? 'Restart required'
                    : 'Saved; rollout required'}
              </strong>
              <p>{model.applyResult.message}</p>
            </div>
          )}
          {model.applyError && (
            <div className={styles.notice} role="alert">
              <strong>Deployment request failed</strong>
              <p>{model.applyError}</p>
              <p>
                Read the saved configuration and runtime status before retrying. Your selection is
                retained.
              </p>
            </div>
          )}
        </section>

        <section className={styles.panel} aria-labelledby="decision-runtime-title">
          <h2 id="decision-runtime-title">Model runtime deployments</h2>
          <p className={styles.muted}>
            Live reports include the shared decision model, specialist overrides, and other router
            model dependencies.
          </p>
          {!model.inventory?.deployments.length && <p>No model runtime deployments reported.</p>}
          <div className={styles.deployments}>
            {model.inventory?.deployments.map((deployment) => (
              <article className={styles.deployment} key={deployment.name}>
                <div className={styles.deploymentTitle}>
                  <h3>{deployment.name}</h3>
                  <span>
                    {deployment.ready && deployment.state === 'ready'
                      ? 'Ready'
                      : deployment.state || 'Not ready'}
                  </span>
                </div>
                {deployment.reason && <p className={styles.notice}>{deployment.reason}</p>}
                <dl className={styles.facts}>
                  <div>
                    <dt>Repository</dt>
                    <dd>{shown(deployment.repo)}</dd>
                  </div>
                  <div>
                    <dt>Revision</dt>
                    <dd>{shown(deployment.revision)}</dd>
                  </div>
                  <div>
                    <dt>Device</dt>
                    <dd>{shown(deployment.device)}</dd>
                  </div>
                  <div>
                    <dt>Engine backend</dt>
                    <dd>{shown(deployment.engine)}</dd>
                  </div>
                  <div>
                    <dt>Profile</dt>
                    <dd>{shown(deployment.profile)}</dd>
                  </div>
                  <div>
                    <dt>Restarts</dt>
                    <dd>{deployment.restarts ?? 'Not reported'}</dd>
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
                  <p className={styles.muted}>Surfaces: {deployment.surfaces.join(', ')}</p>
                ) : null}
                {deployment.heads?.length ? (
                  <p className={styles.muted}>
                    Heads:{' '}
                    {deployment.heads.map((head) => `${head.name} (${head.kind})`).join(', ')}
                  </p>
                ) : null}
              </article>
            ))}
          </div>
        </section>

        <section className={styles.panel} aria-labelledby="decision-consumers-title">
          <h2 id="decision-consumers-title">Bindings and questions</h2>
          {consumers.length ? (
            <div className={styles.tableScroll}>
              <table>
                <thead>
                  <tr>
                    <th>Binding</th>
                    <th>Recipe</th>
                    <th>Deployment</th>
                    <th>State</th>
                  </tr>
                </thead>
                <tbody>
                  {consumers.map((consumer) => (
                    <tr key={`${consumer.recipe ?? 'default'}:${consumer.name}`}>
                      <td>{consumer.name}</td>
                      <td>{consumer.recipe || 'default'}</td>
                      <td>{shown(consumer.metadata?.deployment)}</td>
                      <td>{getRouterModelStateLabel(consumer)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <p>No runtime consumers reported.</p>
          )}
          {scopes.map((scope) => (
            <div className={styles.scope} key={scope.id}>
              <h3>{scope.label}</h3>
              <p>{scope.entrypoints.join(', ') || 'No public model names configured'}</p>
              <p>
                {scope.questions.length
                  ? scope.questions
                      .map((question) => `${question.name} (${question.kind})`)
                      .join(' · ')
                  : 'No custom questions configured.'}
              </p>
            </div>
          ))}
          <h3>Explicit signal overrides</h3>
          {bindings.length ? (
            <dl className={styles.facts}>
              {bindings.map(([name, value]) => (
                <div key={name}>
                  <dt>{name}</dt>
                  <dd>{String(value)}</dd>
                </div>
              ))}
            </dl>
          ) : (
            <p className={styles.muted}>
              No explicit overrides. Built-in signals use the selected model&apos;s defaults.
            </p>
          )}
          <Link to="/config/global-config#global-section-system_models">
            Manage advanced model bindings &rsaquo;
          </Link>
        </section>
      </div>
    </ConfigPageManagerLayout>
  )
}
