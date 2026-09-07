import type { EvaluationGate } from '../../types/evaluationReport'
import { formatDateTime } from '../../utils/dateTime'
import {
  formatMetric,
  formatMetricThreshold,
  gateVerdictPresentation,
} from './evaluationPresentation'
import { GateVerdictBadge } from './EvaluationPrimitives'
import { EvaluationTechnicalDisclosure } from './EvaluationDisclosure'
import styles from './EvaluationGateList.module.css'
import reportStyles from './EvaluationReportLayout.module.css'

const GATE_DISPOSITION_LABELS: Record<EvaluationGate['disposition'], string> = {
  required: 'Required check',
  advisory: 'Recommended check',
  not_applicable: 'Not required for this change',
}

function GateMeasurement({ gate }: { gate: EvaluationGate }) {
  return (
    <>
      {typeof gate.observed === 'number' ? (
        <small>
          Observed {formatMetric({ value: gate.observed, unit: gate.threshold?.unit || '' })}
          {gate.threshold ? ` · ${formatMetricThreshold(gate.threshold)}` : ''}
        </small>
      ) : null}
      {typeof gate.sample_count === 'number' ? <small>{gate.sample_count} samples</small> : null}
      {gate.coverage ? (
        <small>
          Coverage {gate.coverage.evaluated}/{gate.coverage.total} (
          {(gate.coverage.fraction * 100).toFixed(1)}%)
          {gate.coverage.unavailable ? ` · ${gate.coverage.unavailable} not measured` : ''}
          {gate.coverage.confidence_interval
            ? ` · CI ${(gate.coverage.confidence_interval[0] * 100).toFixed(1)}–${(
                gate.coverage.confidence_interval[1] * 100
              ).toFixed(1)}%`
            : ''}
        </small>
      ) : null}
      {gate.evaluated_at ? <small>Evaluated {formatDateTime(gate.evaluated_at)}</small> : null}
    </>
  )
}

interface GateTechnicalDetailsProps {
  gate: EvaluationGate
  capability: string
  supportingRecordCount: number
}

function GateTechnicalDetails({
  gate,
  capability,
  supportingRecordCount,
}: GateTechnicalDetailsProps) {
  if (
    !gate.owner &&
    !gate.rationale &&
    !gate.description &&
    !gate.threshold &&
    !supportingRecordCount
  ) {
    return null
  }
  return (
    <EvaluationTechnicalDisclosure
      className={styles.gateTechnicalDetails}
      summary="Technical details"
      summaryClassName={styles.gateTechnicalSummary}
    >
      {gate.owner ? (
        <p className={styles.gateTechnicalCopy}>
          <strong>Evaluation owner</strong>
          <span>{gate.owner}</span>
        </p>
      ) : null}
      {gate.rationale ? (
        <p className={styles.gateTechnicalCopy}>
          <strong>Recorded rationale</strong>
          <span>{gate.rationale}</span>
        </p>
      ) : null}
      {gate.description ? (
        <p className={styles.gateTechnicalCopy}>
          <strong>Check definition</strong>
          <span>{gate.description}</span>
        </p>
      ) : null}
      {gate.threshold ? (
        <p className={styles.gateTechnicalCopy}>
          <strong>Reported threshold</strong>
          <code>
            {gate.threshold.operator} {gate.threshold.value} {gate.threshold.unit || ''}
          </code>
        </p>
      ) : null}
      {supportingRecordCount ? (
        <ul className={styles.gateReferences} aria-label={`${capability} supporting records`}>
          {gate.evidence_refs.map((reference, index) => (
            <li key={`${index}-${reference}`}>
              <code>{reference}</code>
            </li>
          ))}
        </ul>
      ) : null}
    </EvaluationTechnicalDisclosure>
  )
}

function GateEvidence({ gate, capability }: { gate: EvaluationGate; capability: string }) {
  const supportingRecordCount = gate.evidence_refs.length
  return (
    <div className={styles.gateEvidence}>
      <span className={styles.gateRequirement}>{GATE_DISPOSITION_LABELS[gate.disposition]}</span>
      <GateMeasurement gate={gate} />
      {supportingRecordCount ? (
        <small>
          {supportingRecordCount} supporting {supportingRecordCount === 1 ? 'record' : 'records'}
        </small>
      ) : (
        <small>No supporting records linked.</small>
      )}
      <GateTechnicalDetails
        gate={gate}
        capability={capability}
        supportingRecordCount={supportingRecordCount}
      />
      {gate.disposition === 'required' && gate.verdict === 'unavailable' ? (
        <strong className={styles.unavailableWarning}>
          Release blocked · complete the required evaluation data before this check can pass.
        </strong>
      ) : null}
    </div>
  )
}

function GateRow({ gate }: { gate: EvaluationGate }) {
  const verdict = gateVerdictPresentation(gate)
  const capability = gate.name
  return (
    <article className={styles.gateRow} data-tone={verdict.tone}>
      <div className={styles.gateIdentity}>
        <div>
          <strong>{capability}</strong>
        </div>
        <p className={styles.gateRationale}>{verdict.explanation}</p>
      </div>
      <GateEvidence gate={gate} capability={capability} />
      <span className={styles.gateVerdict}>
        <GateVerdictBadge verdict={gate.verdict} disposition={gate.disposition} />
      </span>
    </article>
  )
}

export default function EvaluationGateList({ gates }: { gates: EvaluationGate[] }) {
  if (gates.length === 0) return <p className={reportStyles.empty}>No release checks apply.</p>
  return (
    <div className={styles.gateList}>
      {gates.map((gate) => (
        <GateRow key={gate.id} gate={gate} />
      ))}
    </div>
  )
}
