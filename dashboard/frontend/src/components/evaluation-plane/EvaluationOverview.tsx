import type {
  EvaluationCatalog,
  EvaluationReport,
  EvaluationRun,
} from '../../types/evaluationPlane'
import { EVALUATION_TRACK_IDS, TRACK_PRESENTATION } from '../../types/evaluationPlane'
import { formatDateTime } from '../../utils/dateTime'
import { effectiveGateVerdict, formatMetric } from './evaluationPresentation'
import type { EvaluationView } from './EvaluationNavigation'
import { GateVerdictBadge, MetricCard, RunStatusBadge } from './EvaluationPrimitives'
import styles from './EvaluationPlane.module.css'

interface EvaluationOverviewProps {
  catalog: EvaluationCatalog
  runs: EvaluationRun[]
  latestReport: EvaluationReport | null
  reportLoading: boolean
  onNavigate: (view: EvaluationView) => void
}

export default function EvaluationOverview({
  catalog,
  runs,
  latestReport,
  reportLoading,
  onNavigate,
}: EvaluationOverviewProps) {
  const running = runs.filter((run) => run.status === 'running').length
  const completed = runs.filter((run) => run.status === 'completed').length
  const failed = runs.filter((run) => run.status === 'failed').length
  const latestRun = runs[0]
  const latestVerdict = latestReport
    ? effectiveGateVerdict(latestReport.summary.verdict, latestReport.gates)
    : null

  return (
    <div className={styles.sectionStack}>
      <section className={styles.callout}>
        <div>
          <span className={styles.eyebrow}>Evidence-first control plane</span>
          <h2>Evaluate the recipe, model pool, and whole system with one run contract.</h2>
          <p>
            Replay deterministic evidence or execute live targets. Every report carries coverage,
            gates, cost ledgers, provenance, artifacts, and architecture recommendations.
          </p>
        </div>
        <div className={styles.calloutActions}>
          <button type="button" className={styles.primaryButton} onClick={() => onNavigate('new')}>
            New experiment
          </button>
          <button
            type="button"
            className={styles.secondaryButton}
            onClick={() => onNavigate('runs')}
          >
            Inspect runs
          </button>
        </div>
      </section>

      <div className={styles.metricGrid}>
        <MetricCard
          label="Catalog suites"
          value={catalog.suites.length}
          detail={`${catalog.targets.length} targets`}
        />
        <MetricCard
          label="Active runs"
          value={running}
          detail={`${runs.length} total`}
          tone={running ? 'positive' : 'neutral'}
        />
        <MetricCard
          label="Completed"
          value={completed}
          detail={latestRun ? `Latest ${formatDateTime(latestRun.created_at)}` : 'No runs yet'}
        />
        <MetricCard
          label="Failed"
          value={failed}
          detail="Terminal pipeline failures"
          tone={failed ? 'negative' : 'neutral'}
        />
      </div>

      <section className={styles.panel}>
        <div className={styles.panelHeader}>
          <div>
            <span className={styles.eyebrow}>Coverage map</span>
            <h2>Eight independent evaluation tracks</h2>
          </div>
          <div className={styles.chips}>
            <span className={styles.schemaBadge}>Contract {catalog.schema_version}</span>
            <span className={styles.schemaBadge}>{catalog.gate_contract_version}</span>
            <span className={styles.schemaBadge}>
              {catalog.change_profiles.length} change profiles
            </span>
          </div>
        </div>
        <div className={styles.trackGrid}>
          {EVALUATION_TRACK_IDS.map((trackID) => {
            const entry = catalog.tracks.find((track) => track.id === trackID)
            return (
              <article key={trackID} className={styles.trackCard}>
                <div className={styles.trackCardHeader}>
                  <h3>{TRACK_PRESENTATION[trackID].label}</h3>
                  <span>{entry ? 'Catalogued' : 'Unavailable'}</span>
                </div>
                <p>{entry?.description || TRACK_PRESENTATION[trackID].description}</p>
                <small>{entry?.metrics.length || 0} declared metrics</small>
              </article>
            )
          })}
        </div>
      </section>

      <section className={styles.panel}>
        <div className={styles.panelHeader}>
          <div>
            <span className={styles.eyebrow}>Latest evidence</span>
            <h2>{latestReport?.run.name || 'No completed report yet'}</h2>
          </div>
          {latestVerdict ? <GateVerdictBadge verdict={latestVerdict} /> : null}
        </div>
        {reportLoading ? (
          <p className={styles.emptyCopy}>Loading the latest report summary…</p>
        ) : latestReport ? (
          <div className={styles.metricGrid}>
            <MetricCard
              label="Quality"
              value={
                latestReport.summary.quality_score === null
                  ? 'Unavailable'
                  : formatMetric({ value: latestReport.summary.quality_score, unit: 'ratio' })
              }
            />
            <MetricCard
              label="P95 latency"
              value={formatMetric({ value: latestReport.summary.latency_p95_ms, unit: 'ms' })}
            />
            <MetricCard
              label="Runtime cost"
              value={formatMetric({ value: latestReport.summary.runtime_cost, unit: 'usd' })}
            />
            <MetricCard
              label="Capacity TCO"
              value={formatMetric({ value: latestReport.summary.capacity_tco, unit: 'usd' })}
            />
          </div>
        ) : (
          <div className={styles.emptyState}>
            <p>Complete a replay or live run to establish the first evidence baseline.</p>
            {latestRun ? <RunStatusBadge status={latestRun.status} /> : null}
          </div>
        )}
      </section>
    </div>
  )
}
