import type { EvaluationCapacityProfile } from '../../types/evaluationCapacityReport'
import type { EvaluationDiagnosticArtifactIssue } from '../../types/evaluationReportDiagnostics'
import { formatMetric } from './evaluationPresentation'
import { EvaluationTag } from './EvaluationPrimitives'
import EvaluationReportCapacityTable from './EvaluationReportCapacityTable'
import EvaluationReportDiagnosticIssue from './EvaluationReportDiagnosticIssue'
import styles from './EvaluationReportDiagnostics.module.css'
import reportStyles from './EvaluationReportLayout.module.css'

interface EvaluationReportCapacityProps {
  capacityProfile: EvaluationCapacityProfile | null
  issue: EvaluationDiagnosticArtifactIssue | null
}

const CAPACITY_FAILURE_LABELS = {
  required_concurrency: 'Required concurrency was not supported',
  warmup_errors: 'Warmup produced request errors',
  latency_p95: 'p95 latency exceeded its bound',
  measurement_cluster_coverage: 'Too few independent measurement windows were observed',
  error_rate_cluster_stability: 'Error rates varied too much between measurement windows',
  error_rate_upper_bound: 'A measurement window’s 95% error-rate bound exceeded its budget',
  throughput: 'Throughput missed its minimum',
  throughput_scaling: 'Throughput scaling reached saturation',
  throughput_stability: 'Throughput varied beyond the recorded stability limit',
  latency_stability: 'p95 latency varied beyond the recorded stability limit',
} as const

function CapacitySummary({ profile }: { profile: EvaluationCapacityProfile }) {
  return (
    <div className={styles.diagnosticSummary}>
      <div>
        <span>Required concurrency</span>
        <strong>{profile.slo.required_concurrency}</strong>
      </div>
      <div>
        <span>Supported concurrency</span>
        <strong>{profile.assessment.qualified_concurrency ?? '—'}</strong>
      </div>
      <div>
        <span>Capacity above objective</span>
        <strong>
          {profile.assessment.slo_headroom > 0 ? '+' : ''}
          {profile.assessment.slo_headroom}
        </strong>
      </div>
      <div>
        <span>Saturation boundary</span>
        <strong>{profile.assessment.saturation_concurrency ?? 'Not observed'}</strong>
      </div>
    </div>
  )
}

function CapacitySLOContract({ profile }: { profile: EvaluationCapacityProfile }) {
  return (
    <div className={styles.capacitySLOContract} aria-label="Recorded capacity objective">
      <div>
        <span>p95 latency</span>
        <strong>≤ {formatMetric({ value: profile.slo.max_latency_p95_ms, unit: 'ms' })}</strong>
      </div>
      <div>
        <span>Error rate</span>
        <strong>≤ {(profile.slo.max_error_rate * 100).toFixed(2)}%</strong>
      </div>
      <div>
        <span>Throughput at target</span>
        <strong>
          ≥ {formatMetric({ value: profile.slo.min_throughput_rps, unit: 'requests/s' })}
        </strong>
      </div>
      <div>
        <span>Scaling efficiency</span>
        <strong>≥ {(profile.slo.min_throughput_scaling_efficiency * 100).toFixed(1)}%</strong>
      </div>
    </div>
  )
}

function CapacityProtocolContract({ profile }: { profile: EvaluationCapacityProfile }) {
  return (
    <div className={styles.capacityProtocolContract} aria-label="Recorded capacity load plan">
      <div>
        <span>Closed-loop ladder</span>
        <strong>
          {profile.protocol.concurrency_levels.map((level) => `c${level}`).join(' → ')}
        </strong>
      </div>
      <div>
        <span>Warmup per level</span>
        <strong>{profile.protocol.warmup_request_multiplier} × concurrency requests</strong>
      </div>
      <div>
        <span>Measurement window</span>
        <strong>
          {profile.protocol.measurement_requests_per_repetition} requests ×{' '}
          {profile.protocol.repetitions_per_level} repetitions
        </strong>
      </div>
      <div>
        <span>Independent-window controls</span>
        <strong>
          ≥ {profile.protocol.minimum_measurement_clusters_per_level} windows · error spread ≤{' '}
          {(profile.protocol.max_error_rate_cluster_range * 100).toFixed(0)}% ·{' '}
          {(profile.protocol.confidence_level * 100).toFixed(0)}% bound
        </strong>
      </div>
      <div>
        <span>Performance stability</span>
        <strong>
          throughput variation ≤{' '}
          {(profile.protocol.max_throughput_cv * 100).toFixed(0)}% · p95 variation ≤{' '}
          {(profile.protocol.max_latency_p95_cv * 100).toFixed(0)}%
        </strong>
      </div>
    </div>
  )
}

function CapacityFailureReasons({ profile }: { profile: EvaluationCapacityProfile }) {
  if (!profile.assessment.failure_reasons.length) return null
  return (
    <div className={styles.capacityFailureReasons} role="status">
      <strong>Why the envelope failed</strong>
      <ul>
        {profile.assessment.failure_reasons.map((reason) => (
          <li key={reason}>{CAPACITY_FAILURE_LABELS[reason]}</li>
        ))}
      </ul>
    </div>
  )
}

function CapacityEvidence({ profile }: { profile: EvaluationCapacityProfile }) {
  return (
    <>
      <CapacitySummary profile={profile} />
      <CapacitySLOContract profile={profile} />
      <CapacityProtocolContract profile={profile} />
      <CapacityFailureReasons profile={profile} />
      <EvaluationReportCapacityTable levels={profile.levels} />
    </>
  )
}

export default function EvaluationReportCapacity({
  capacityProfile,
  issue,
}: EvaluationReportCapacityProps) {
  if (!capacityProfile && !issue) return null
  return (
    <section className={styles.diagnosticArtifact} aria-labelledby="diagnostic-capacity-title">
      <div className={reportStyles.subsectionHeader}>
        <div>
          <h4 id="diagnostic-capacity-title">Capacity envelope</h4>
          <p>Observed live load against the service objective selected for this run.</p>
        </div>
        {capacityProfile ? (
          <EvaluationTag
            tone={capacityProfile.assessment.verdict === 'pass' ? 'positive' : 'warning'}
          >
            {capacityProfile.assessment.verdict === 'pass'
              ? 'Capacity target passed'
              : 'Capacity target blocked'}
          </EvaluationTag>
        ) : null}
      </div>
      {issue ? (
        <EvaluationReportDiagnosticIssue label="Capacity profile" issue={issue} />
      ) : capacityProfile ? (
        <CapacityEvidence profile={capacityProfile} />
      ) : null}
    </section>
  )
}
