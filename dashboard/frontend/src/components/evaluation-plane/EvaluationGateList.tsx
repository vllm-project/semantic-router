import type { EvaluationGate } from '../../types/evaluationPlane'
import { formatDateTime } from '../../utils/dateTime'
import { formatMetric } from './evaluationPresentation'
import { GateVerdictBadge } from './EvaluationPrimitives'
import styles from './EvaluationReport.module.css'

export default function EvaluationGateList({ gates }: { gates: EvaluationGate[] }) {
  if (gates.length === 0) return <p className={styles.empty}>No gates were declared.</p>

  return (
    <div className={styles.gateList}>
      {gates.map((gate) => (
        <article key={gate.id} className={styles.gateRow}>
          <div className={styles.gateIdentity}>
            <div>
              <strong>{gate.name}</strong>
              <code>{gate.id}</code>
            </div>
            <p>{gate.description || gate.rationale || 'No gate rationale was recorded.'}</p>
          </div>
          <div className={styles.gateEvidence}>
            <span>{gate.disposition.replace('_', ' ')}</span>
            {typeof gate.observed === 'number' ? (
              <small>
                Observed {formatMetric({ value: gate.observed, unit: gate.threshold?.unit || '' })}
                {gate.threshold
                  ? ` · ${gate.threshold.operator} ${formatMetric({ value: gate.threshold.value, unit: gate.threshold.unit || '' })}`
                  : ''}
              </small>
            ) : null}
            <small>
              Profile <code>{gate.change_profile}</code> · contract{' '}
              <code>{gate.contract_version}</code>
            </small>
            {typeof gate.sample_count === 'number' ? <small>N = {gate.sample_count}</small> : null}
            {gate.coverage ? (
              <small>
                Coverage {gate.coverage.evaluated}/{gate.coverage.total} (
                {(gate.coverage.fraction * 100).toFixed(1)}%)
                {gate.coverage.unavailable ? ` · ${gate.coverage.unavailable} unavailable` : ''}
                {gate.coverage.confidence_interval
                  ? ` · CI ${(gate.coverage.confidence_interval[0] * 100).toFixed(1)}–${(
                      gate.coverage.confidence_interval[1] * 100
                    ).toFixed(1)}%`
                  : ''}
              </small>
            ) : null}
            {gate.owner ? <small>Owner {gate.owner}</small> : null}
            {gate.evaluated_at ? (
              <small>Evaluated {formatDateTime(gate.evaluated_at)}</small>
            ) : null}
            {gate.evidence_refs.length ? (
              <ul className={styles.gateReferences} aria-label={`${gate.id} evidence references`}>
                {gate.evidence_refs.map((reference, index) => (
                  <li key={`${index}-${reference}`}>
                    <code>{reference}</code>
                  </li>
                ))}
              </ul>
            ) : (
              <small>No evidence references recorded.</small>
            )}
            {gate.disposition === 'required' && gate.verdict === 'unavailable' ? (
              <strong className={styles.unavailableWarning}>
                Required gate is not satisfied: unavailable evidence never counts as pass.
              </strong>
            ) : null}
          </div>
          <GateVerdictBadge verdict={gate.verdict} />
        </article>
      ))}
    </div>
  )
}
