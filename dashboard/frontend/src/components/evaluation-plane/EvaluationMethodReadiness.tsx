import type { EvaluationCatalog, EvaluationTrackId } from '../../types/evaluationPlane'
import { EVALUATION_TRACK_IDS } from '../../types/evaluationPlane'
import EvaluationDisclosure from './EvaluationDisclosure'
import { TRACK_PRESENTATION } from './evaluationTrackPresentation'
import EvaluationIssueDetails from './EvaluationIssueDetails'
import { EvaluationTag } from './EvaluationPrimitives'
import type {
  EvaluationMethodReadinessEntry,
  EvaluationMethodReadinessStatus,
} from './evaluationMethodReadinessModel'
import {
  EVALUATION_METHOD_EVIDENCE_SOURCE_LABELS,
  EVALUATION_METHOD_STATUS_LABELS,
  evaluationMethodCapabilityLabel,
  evaluationMethodSetupGuidance,
  evaluationMethodTechnicalDetails,
} from './evaluationMethodReadinessPresentation'
import useEvaluationMethodReadiness from './useEvaluationMethodReadiness'
import styles from './EvaluationMethodReadiness.module.css'
import planeStyles from './EvaluationPlane.module.css'
import tableStyles from './EvaluationReportTable.module.css'

function MethodReadinessHeader({
  counts,
}: {
  counts: Record<EvaluationMethodReadinessStatus, number>
}) {
  return (
    <header className={planeStyles.surfaceHeader}>
      <div>
        <span className={planeStyles.eyebrow}>Evaluation methods</span>
        <h2 id="evaluation-methods-title">Available benchmark capabilities</h2>
        <p>
          See what each method measures, where its data comes from, and which release checks it can
          support. Imported results remain diagnostic until a managed run verifies the benchmark
          execution.
        </p>
      </div>
      <div className={styles.methodSummary} aria-label="Method readiness summary">
        <span>
          <strong>{counts.ready}</strong> ready
        </span>
        <span>
          <strong>{counts.setup_required}</strong> need setup
        </span>
      </div>
    </header>
  )
}

interface MethodFiltersProps {
  query: string
  track: EvaluationTrackId | 'all'
  status: EvaluationMethodReadinessStatus | 'all'
  visibleCount: number
  methodCount: number
  onQueryChange: (value: string) => void
  onTrackChange: (value: EvaluationTrackId | 'all') => void
  onStatusChange: (value: EvaluationMethodReadinessStatus | 'all') => void
}

function MethodFilters({
  query,
  track,
  status,
  visibleCount,
  methodCount,
  onQueryChange,
  onTrackChange,
  onStatusChange,
}: MethodFiltersProps) {
  return (
    <div className={styles.methodFilters}>
      <label className={styles.methodSearch}>
        <span>Search benchmark methods</span>
        <input
          type="search"
          aria-label="Search evaluation methods"
          value={query}
          placeholder="Benchmark, evaluation area, or release check…"
          onChange={(event) => onQueryChange(event.target.value)}
        />
      </label>
      <label>
        <span>Evaluation area</span>
        <select
          aria-label="Method evaluation area filter"
          value={track}
          onChange={(event) => onTrackChange(event.target.value as EvaluationTrackId | 'all')}
        >
          <option value="all">All areas</option>
          {EVALUATION_TRACK_IDS.map((trackID) => (
            <option key={trackID} value={trackID}>
              {TRACK_PRESENTATION[trackID].label}
            </option>
          ))}
        </select>
      </label>
      <label>
        <span>Readiness</span>
        <select
          aria-label="Method readiness filter"
          value={status}
          onChange={(event) =>
            onStatusChange(event.target.value as EvaluationMethodReadinessStatus | 'all')
          }
        >
          <option value="all">All states</option>
          {Object.entries(EVALUATION_METHOD_STATUS_LABELS).map(([value, label]) => (
            <option key={value} value={value}>
              {label}
            </option>
          ))}
        </select>
      </label>
      <span className={styles.resultCount} role="status">
        Showing {visibleCount} of {methodCount} methods
      </span>
    </div>
  )
}

function MethodReadinessRow({ entry }: { entry: EvaluationMethodReadinessEntry }) {
  const { method, qualifiedGateNames, readiness, suiteID, suiteName, revision, executors } = entry
  return (
    <tr>
      <th scope="row">
        <strong>{suiteName}</strong>
        <span>{evaluationMethodCapabilityLabel(method, qualifiedGateNames)}</span>
      </th>
      <td>{TRACK_PRESENTATION[method.track_id].label}</td>
      <td>{EVALUATION_METHOD_EVIDENCE_SOURCE_LABELS[method.evidence_source]}</td>
      <td>
        {qualifiedGateNames.length > 0 ? qualifiedGateNames.join(' · ') : 'Exploratory only'}
      </td>
      <td>
        <div className={styles.methodReadiness}>
          <EvaluationTag tone={readiness === 'ready' ? 'info' : 'warning'}>
            {EVALUATION_METHOD_STATUS_LABELS[readiness]}
          </EvaluationTag>
          <small className={styles.methodGuidance}>
            {evaluationMethodSetupGuidance(method, readiness)}
          </small>
          <EvaluationIssueDetails
            className={styles.methodTechnicalDetails}
            issues={evaluationMethodTechnicalDetails({ method, suiteID, revision, executors })}
          />
        </div>
      </td>
    </tr>
  )
}

function MethodReadinessTable({ methods }: { methods: EvaluationMethodReadinessEntry[] }) {
  return (
    <div
      className={`${tableStyles.tableScroll} ${styles.methodTableFrame}`}
      role="region"
      tabIndex={0}
      aria-label="Scrollable evaluation method readiness"
    >
      <table className={`${tableStyles.table} ${tableStyles.tableReadiness} ${styles.methodTable}`}>
        <caption>Available evaluation methods and setup readiness</caption>
        <thead>
          <tr>
            <th scope="col">Benchmark method</th>
            <th scope="col">Evaluation area</th>
            <th scope="col">Evidence source</th>
            <th scope="col">Release checks</th>
            <th scope="col">Readiness</th>
          </tr>
        </thead>
        <tbody>
          {methods.map((entry) => (
            <MethodReadinessRow key={`${entry.suiteID}:${entry.method.id}`} entry={entry} />
          ))}
          {methods.length === 0 ? (
            <tr>
              <td className={styles.methodEmpty} colSpan={5}>
                No methods match these filters.
              </td>
            </tr>
          ) : null}
        </tbody>
      </table>
    </div>
  )
}

export default function EvaluationMethodReadiness({ catalog }: { catalog: EvaluationCatalog }) {
  const model = useEvaluationMethodReadiness(catalog)

  return (
    <section className={planeStyles.surface} aria-labelledby="evaluation-methods-title">
      <MethodReadinessHeader counts={model.counts} />
      <EvaluationDisclosure
        className={styles.methodDisclosure}
        summaryClassName={styles.methodDisclosureSummary}
        summary={
          <span>
            <strong>Browse benchmark methods</strong>
            <small>
              Search all {model.methods.length} methods when you need implementation and setup
              details.
            </small>
          </span>
        }
      >
        <div className={styles.methodCatalog}>
          <MethodFilters
            query={model.query}
            track={model.track}
            status={model.status}
            visibleCount={model.visibleMethods.length}
            methodCount={model.methods.length}
            onQueryChange={model.setQuery}
            onTrackChange={model.setTrack}
            onStatusChange={model.setStatus}
          />
          <MethodReadinessTable methods={model.visibleMethods} />
        </div>
      </EvaluationDisclosure>
    </section>
  )
}
