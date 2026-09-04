import type { EvaluationReport } from '../../types/evaluationPlane'
import { TRACK_PRESENTATION } from '../../types/evaluationPlane'
import { formatDateTime } from '../../utils/dateTime'
import {
  getEvaluationArtifactURL,
  isDownloadableEvaluationArtifact,
} from '../../utils/evaluationPlaneApi'
import EvaluationGateList from './EvaluationGateList'
import { effectiveGateVerdict, formatMetric } from './evaluationPresentation'
import {
  CoverageBar,
  GateVerdictBadge,
  MetricCard,
  MetricGrid,
  RunStatusBadge,
} from './EvaluationPrimitives'
import styles from './EvaluationReport.module.css'

export default function EvaluationReportView({ report }: { report: EvaluationReport }) {
  const summary = report.summary
  const gateContractVersion = report.gates[0]?.contract_version || 'unavailable'
  const requiredGates = report.gates.filter((gate) => gate.disposition === 'required')
  const requiredPassed = requiredGates.filter((gate) => gate.verdict === 'pass').length
  const requiredFailed = requiredGates.filter((gate) => gate.verdict === 'fail').length
  const requiredUnavailable = requiredGates.filter((gate) => gate.verdict === 'unavailable').length
  const promotionVerdict = effectiveGateVerdict(summary.verdict, report.gates)
  return (
    <div className={styles.report}>
      <section className={styles.reportHero}>
        <div>
          <span className={styles.eyebrow}>Evidence report · {report.schema_version}</span>
          <h2>{report.run.name}</h2>
          <p>{report.run.description || 'No experiment description.'}</p>
          <div className={styles.heroBadges}>
            <RunStatusBadge status={report.run.status} />
            <GateVerdictBadge verdict={promotionVerdict} />
            <span>{report.run.evidence_level}</span>
            <span>{report.run.mode}</span>
            <span>Profile {report.run.change_profile}</span>
            <span>Gate contract {gateContractVersion}</span>
          </div>
        </div>
        <CoverageBar coverage={summary.coverage} />
      </section>

      <div className={styles.summaryGrid}>
        <MetricCard
          label="Quality"
          value={formatMetric({ value: summary.quality_score, unit: 'ratio' })}
          detail="Aggregate task quality"
        />
        <MetricCard
          label="P95 latency"
          value={formatMetric({ value: summary.latency_p95_ms, unit: 'ms' })}
          detail="End-to-end request latency"
        />
        <MetricCard
          label="Runtime cost"
          value={formatMetric({ value: summary.runtime_cost, unit: 'usd' })}
          detail="Serving execution ledger"
        />
        <MetricCard
          label="Capacity TCO"
          value={formatMetric({ value: summary.capacity_tco, unit: 'usd' })}
          detail="Capacity planning ledger"
        />
        <MetricCard
          label="Required gates"
          value={`${requiredPassed}/${requiredGates.length} passed`}
          detail={`${requiredFailed} failed · ${requiredUnavailable} unavailable`}
          tone={requiredFailed ? 'negative' : requiredUnavailable ? 'warning' : 'positive'}
        />
        <MetricCard
          label="Unavailable"
          value={summary.unavailable_gates}
          detail="Never counted as pass"
          tone={summary.unavailable_gates ? 'warning' : 'neutral'}
        />
      </div>

      <section className={styles.section}>
        <div className={styles.sectionHeader}>
          <div>
            <span className={styles.eyebrow}>Decomposition</span>
            <h3>Track evidence</h3>
          </div>
          <span>{report.tracks.length} tracks</span>
        </div>
        <div className={styles.trackList}>
          {report.tracks.map((track) => (
            <details
              key={track.track_id}
              className={styles.track}
              open={track.status === 'failed' || track.status === 'unavailable'}
            >
              <summary>
                <div>
                  <strong>{TRACK_PRESENTATION[track.track_id].label}</strong>
                  <span>{track.summary}</span>
                </div>
                <div>
                  <span>{track.evidence_level}</span>
                  <RunStatusBadge status={track.status} />
                </div>
              </summary>
              <div className={styles.trackBody}>
                <CoverageBar coverage={track.coverage} />
                {track.error ? (
                  <div className={styles.error} role="alert">
                    {track.error}
                  </div>
                ) : null}
                <MetricGrid metrics={track.metrics} />
                <EvaluationGateList gates={track.gates} />
              </div>
            </details>
          ))}
        </div>
      </section>

      <section className={styles.section}>
        <div className={styles.sectionHeader}>
          <div>
            <span className={styles.eyebrow}>Decision boundary</span>
            <h3>Promotion gates</h3>
          </div>
          <GateVerdictBadge verdict={promotionVerdict} />
        </div>
        <EvaluationGateList gates={report.gates} />
      </section>

      <section className={styles.section}>
        <div className={styles.sectionHeader}>
          <div>
            <span className={styles.eyebrow}>All aggregates</span>
            <h3>Metrics</h3>
          </div>
          <span>{report.metrics.length} metrics</span>
        </div>
        <MetricGrid metrics={report.metrics} />
      </section>

      <section className={styles.section}>
        <div className={styles.sectionHeader}>
          <div>
            <span className={styles.eyebrow}>Three ledgers</span>
            <h3>Cost accounting</h3>
          </div>
        </div>
        <div className={styles.ledgerGrid}>
          {Object.entries(report.costs).map(([name, ledger]) => (
            <article key={name}>
              <span>{name.replace('_', ' ')}</span>
              <strong>
                {formatMetric({ value: ledger.amount, unit: ledger.currency.toLowerCase() })}
              </strong>
              <small>
                {ledger.input_tokens || ledger.output_tokens
                  ? `${ledger.input_tokens || 0} input · ${ledger.output_tokens || 0} output tokens`
                  : 'Token accounting unavailable'}
                {ledger.gpu_seconds ? ` · ${ledger.gpu_seconds.toFixed(1)} GPU seconds` : ''}
                {ledger.energy_kwh ? ` · ${ledger.energy_kwh.toFixed(2)} kWh` : ''}
              </small>
            </article>
          ))}
        </div>
      </section>

      <section className={styles.twoColumn}>
        <div className={styles.section}>
          <div className={styles.sectionHeader}>
            <div>
              <span className={styles.eyebrow}>Architecture feedback</span>
              <h3>Recommendations</h3>
            </div>
          </div>
          {report.recommendations.length ? (
            <ol className={styles.recommendations}>
              {report.recommendations.map((item, index) => (
                <li key={`${index}-${item}`}>{item}</li>
              ))}
            </ol>
          ) : (
            <p className={styles.empty}>No architecture recommendations were generated.</p>
          )}
        </div>
        <div className={styles.section}>
          <div className={styles.sectionHeader}>
            <div>
              <span className={styles.eyebrow}>Reproducibility</span>
              <h3>Provenance</h3>
            </div>
          </div>
          <dl className={styles.provenance}>
            <div>
              <dt>Generated</dt>
              <dd>{formatDateTime(report.provenance.generated_at)}</dd>
            </div>
            <div>
              <dt>Target</dt>
              <dd>{report.provenance.target_id}</dd>
            </div>
            <div>
              <dt>Change profile</dt>
              <dd>
                <code>{report.run.change_profile}</code>
              </dd>
            </div>
            <div>
              <dt>Gate contract</dt>
              <dd>
                <code>{gateContractVersion}</code>
              </dd>
            </div>
            <div>
              <dt>Seed</dt>
              <dd>{report.provenance.seed}</dd>
            </div>
            <div>
              <dt>Code revision</dt>
              <dd>
                <code>{report.provenance.code_revision || '-'}</code>
              </dd>
            </div>
            <div>
              <dt>Benchmark revisions</dt>
              <dd>
                {Object.entries(report.provenance.benchmark_revisions || {}).length ? (
                  Object.entries(report.provenance.benchmark_revisions || {}).map(
                    ([benchmark, revision]) => (
                      <span key={benchmark}>
                        {benchmark}: <code>{revision}</code>
                      </span>
                    ),
                  )
                ) : (
                  <code>-</code>
                )}
              </dd>
            </div>
            <div>
              <dt>Workload snapshot</dt>
              <dd>
                <code>{report.provenance.workload_snapshot_digest || '-'}</code>
              </dd>
            </div>
            <div>
              <dt>Policy snapshot</dt>
              <dd>
                <code>{report.provenance.policy_snapshot_digest || '-'}</code>
              </dd>
            </div>
            <div>
              <dt>Policy binding</dt>
              <dd>
                <code>{report.provenance.binding_snapshot_digest || '-'}</code>
              </dd>
            </div>
            <div>
              <dt>Pool snapshot</dt>
              <dd>
                <code>{report.provenance.pool_snapshot_digest || '-'}</code>
              </dd>
            </div>
            <div>
              <dt>Environment</dt>
              <dd>
                <code>{report.provenance.environment_snapshot_digest || '-'}</code>
              </dd>
            </div>
            <div>
              <dt>Redaction</dt>
              <dd>{report.provenance.redaction_policy || '-'}</dd>
            </div>
          </dl>
        </div>
      </section>

      <section className={styles.section}>
        <div className={styles.sectionHeader}>
          <div>
            <span className={styles.eyebrow}>Evidence outputs</span>
            <h3>Artifacts</h3>
          </div>
          <span>{report.artifacts.length} objects</span>
        </div>
        {report.artifacts.length ? (
          <div className={styles.artifactList}>
            {report.artifacts.map((artifact) => (
              <article key={artifact.id}>
                <div>
                  <strong>{artifact.name}</strong>
                  <span>
                    {artifact.kind} · {artifact.media_type || 'unknown media type'}
                  </span>
                </div>
                {isDownloadableEvaluationArtifact(artifact) ? (
                  <a
                    href={getEvaluationArtifactURL(report.run.id, artifact.id)}
                    aria-label={`Download ${artifact.name}`}
                  >
                    Download
                  </a>
                ) : (
                  <code>{artifact.digest || artifact.id}</code>
                )}
              </article>
            ))}
          </div>
        ) : (
          <p className={styles.empty}>No report artifacts were recorded.</p>
        )}
      </section>
    </div>
  )
}
