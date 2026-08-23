import {
  listManagedRecipeScopes,
  type ManagedRoutingSummary,
} from '../utils/managedRoutingSnapshot'
import { countProjectionsInProfile, countSignalsInProfile } from '../utils/routingScopes'
import styles from './DashboardRoutingProfiles.module.css'

interface DashboardRoutingProfilesProps {
  config: ManagedRoutingSummary
  onOpenTopology: (scopeId: string) => void
}

export default function DashboardRoutingProfiles({
  config,
  onOpenTopology,
}: DashboardRoutingProfilesProps) {
  const scopes = listManagedRecipeScopes(config)
  if (scopes.length === 0) return null

  return (
    <section className={styles.section}>
      <div className={styles.header}>
        <div>
          <span className={styles.eyebrow}>Mixture-of-Models</span>
          <h2>Routing objectives</h2>
          <p>Request-facing entrypoints mapped to isolated routing recipes.</p>
        </div>
        <span className={styles.count}>{scopes.length} profiles</span>
      </div>
      <div className={styles.grid}>
        {scopes.map((scope) => {
          const signals = countSignalsInProfile(scope.document).total
          const projections = countProjectionsInProfile(scope.document)
          const decisions = Array.isArray(scope.document.decisions)
            ? scope.document.decisions.length
            : 0
          return (
            <article key={scope.id} className={styles.card}>
              <div className={styles.cardHeader}>
                <div>
                  <h3>{scope.label}</h3>
                  <p>{scope.description || 'Isolated semantic routing profile.'}</p>
                </div>
                <button
                  type="button"
                  onClick={() => onOpenTopology(scope.id)}
                  aria-label={`View ${scope.label} topology`}
                >
                  View topology
                </button>
              </div>
              {scope.entrypointModelNames.length > 0 ? (
                <div className={styles.aliases}>
                  {scope.entrypointModelNames.map((model) => (
                    <span key={model}>{model}</span>
                  ))}
                </div>
              ) : null}
              <dl className={styles.metrics}>
                <div>
                  <dt>Signals</dt>
                  <dd>{signals}</dd>
                </div>
                <div>
                  <dt>Projections</dt>
                  <dd>{projections}</dd>
                </div>
                <div>
                  <dt>Decisions</dt>
                  <dd>{decisions}</dd>
                </div>
              </dl>
            </article>
          )
        })}
      </div>
    </section>
  )
}
