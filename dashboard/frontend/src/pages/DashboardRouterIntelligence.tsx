import RouterModelInventory from '../components/RouterModelInventory'
import { formatRouterModelLabel } from '../components/routerModelPresentation'
import {
  getLoadedModelCount,
  getTotalKnownModelCount,
  type RouterModelInfo,
  type RouterModelsInfo,
} from '../utils/routerRuntime'
import type { RouterConfig } from './dashboardPageTypes'
import {
  buildIntelligenceRoutingScopes,
  getDecisionRuntimeSummary,
} from './dashboardRouterIntelligenceSupport'
import styles from './DashboardRouterIntelligence.module.css'

interface DashboardRouterIntelligenceProps {
  config: RouterConfig | null
  modelsInfo?: RouterModelsInfo | null
  onConfigure: () => void
  onOpenStatus: () => void
  onSelectModel: (model: RouterModelInfo) => void
}

const runtimeLabels = {
  ready: 'Shared runtime ready',
  attention: 'Runtime needs attention',
  unreported: 'Runtime not reported',
  specialists: 'Specialist models',
}

export default function DashboardRouterIntelligence({
  config,
  modelsInfo,
  onConfigure,
  onOpenStatus,
  onSelectModel,
}: DashboardRouterIntelligenceProps) {
  const runtime = config ? getDecisionRuntimeSummary(config, modelsInfo) : null
  const scopes = config ? buildIntelligenceRoutingScopes(config) : []
  const knownModels = getTotalKnownModelCount(modelsInfo)

  return (
    <section className={styles.card} aria-labelledby="router-intelligence-title">
      <div className={styles.header}>
        <h2 id="router-intelligence-title">Router Intelligence</h2>
        <button type="button" className={styles.action} onClick={onConfigure}>
          Configure decision model &rsaquo;
        </button>
      </div>

      {runtime ? (
        <div className={styles.model} data-testid="decision-model-overview">
          <div>
            <span className={styles.eyebrow}>Configured decision model</span>
            <h3>{runtime.model}</h3>
            <p>
              {runtime.model === 'Vela-1.0'
                ? 'Specialist models answer built-in signals. Custom questions need their own deployment.'
                : 'Default model for learned signals, custom questions and decision selectors.'}
            </p>
          </div>
          <div className={styles.runtime}>
            <span className={runtime.state === 'ready' ? styles.ready : styles.pending}>
              {runtimeLabels[runtime.state]}
            </span>
            {runtime.bindings > 0 && (
              <span>
                {runtime.resources} runtime{runtime.resources === 1 ? '' : 's'} · {runtime.bindings}{' '}
                reported binding{runtime.bindings === 1 ? '' : 's'}
              </span>
            )}
          </div>
        </div>
      ) : (
        <p className={styles.unavailable}>Decision model configuration is unavailable.</p>
      )}

      {scopes.length > 0 && (
        <div className={styles.scopes}>
          <h3 className={styles.subheading}>Configured routing</h3>
          {scopes.map((scope) => {
            const signalCount = scope.signals.reduce(
              (total, group) => total + group.names.length,
              0,
            )
            return (
              <details key={scope.id} className={styles.scope}>
                <summary>
                  <span className={styles.scopeIdentity}>
                    <strong>{scope.label}</strong>
                    {scope.entrypoints.length > 0 && <span>{scope.entrypoints.join(', ')}</span>}
                  </span>
                  <span className={styles.counts}>
                    {scope.questions.length} questions · {signalCount} other signals ·{' '}
                    {scope.projections.length} projections
                  </span>
                </summary>
                <div className={styles.scopeContent}>
                  {scope.questions.length > 0 && (
                    <div className={styles.group}>
                      <h4>Custom questions</h4>
                      <div className={styles.chips}>
                        {scope.questions.map(({ name, kind }) => (
                          <span key={name}>
                            <strong>{name}</strong>
                            <span>{kind}</span>
                          </span>
                        ))}
                      </div>
                    </div>
                  )}
                  {scope.signals.map(({ type, names }) => (
                    <div className={styles.group} key={type}>
                      <h4>{formatRouterModelLabel(type)} signals</h4>
                      <div className={styles.chips}>
                        {names.map((name) => (
                          <span key={name}>{name}</span>
                        ))}
                      </div>
                    </div>
                  ))}
                  {scope.projections.length > 0 && (
                    <div className={styles.group}>
                      <h4>Projections</h4>
                      <div className={styles.chips}>
                        {scope.projections.map(({ name, kind }) => (
                          <span key={`${kind}:${name}`}>
                            <strong>{name}</strong>
                            <span>{kind}</span>
                          </span>
                        ))}
                      </div>
                    </div>
                  )}
                  {!scope.questions.length && !signalCount && !scope.projections.length && (
                    <p>No signals or projections configured.</p>
                  )}
                </div>
              </details>
            )
          })}
        </div>
      )}

      <div className={styles.inventoryHeader}>
        <h3 className={styles.subheading}>Runtime inventory</h3>
        {knownModels > 0 && (
          <span>
            {getLoadedModelCount(modelsInfo)}/{knownModels} runtimes ready
          </span>
        )}
        <button type="button" className={styles.action} onClick={onOpenStatus}>
          Status &rsaquo;
        </button>
      </div>
      <RouterModelInventory
        mode="preview"
        modelsInfo={modelsInfo}
        emptyMessage="No runtime inventory reported yet."
        onSelectModel={onSelectModel}
      />
    </section>
  )
}
