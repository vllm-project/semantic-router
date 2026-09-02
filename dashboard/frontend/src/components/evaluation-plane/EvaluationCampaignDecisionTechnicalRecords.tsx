import type {
  EvaluationCampaignEvidenceAnchor,
  EvaluationCampaignEvidenceBindingRole,
  EvaluationCampaignGate,
} from '../../types/evaluationCampaign'
import { EvaluationTag } from './EvaluationPrimitives'
import { CopyableValue } from './EvaluationCampaignDecisionTechnicalFields'
import { shortIdentity } from './evaluationTechnicalFields'
import sectionStyles from './EvaluationCampaignDecisionTechnicalDetails.module.css'
import styles from './EvaluationCampaignDecisionTechnicalRecords.module.css'

const BINDING_ROLE_LABELS = {
  evidence: 'Evaluation run',
  baseline: 'Baseline run',
  candidate: 'Candidate run',
  reference: 'Reference run',
  live: 'Fresh live run',
} satisfies Record<EvaluationCampaignEvidenceBindingRole, string>

function evidenceReferenceLabel(reference: string): string {
  const prefix = reference.split(':', 1)[0]
  switch (prefix) {
    case 'run':
      return 'Run record'
    case 'manifest-semantic':
      return 'Frozen configuration identity'
    case 'manifest-artifact':
      return 'Run manifest'
    case 'report':
      return 'Sealed report'
    case 'private-receipt':
      return 'Private execution receipt'
    case 'candidate-subject':
      return 'Candidate configuration'
    case 'execution-attestation':
      return 'Server execution receipt'
    case 'campaign-fidelity':
      return 'Live consistency receipt'
    default:
      return 'Supporting evaluation record'
  }
}

export function RecordedServiceNarrative({
  summary,
  gates,
  recommendations,
}: {
  summary: string
  gates: EvaluationCampaignGate[]
  recommendations: string[]
}) {
  return (
    <section
      className={sectionStyles.technicalGroup}
      aria-labelledby="campaign-service-narrative-title"
    >
      <div className={sectionStyles.technicalGroupHeader}>
        <h4 id="campaign-service-narrative-title">Recorded service narrative</h4>
        <p>Verbatim decision text retained for audit and independent verification.</p>
      </div>
      <div className={styles.narrative}>
        <div className={styles.narrativeBlock}>
          <h5>Decision summary</h5>
          <p>{summary}</p>
        </div>
        <div className={styles.narrativeBlock}>
          <h5>Recorded check rationale</h5>
          <ul className={styles.gateNarratives}>
            {gates.map((gate) => (
              <li key={gate.id} data-check-id={gate.id}>
                <strong>{gate.name}</strong>
                <p>{gate.rationale}</p>
                {gate.threshold ? (
                  <p>
                    Reported threshold:{' '}
                    <code>
                      {gate.threshold.operator} {gate.threshold.value} {gate.threshold.unit || ''}
                    </code>
                  </p>
                ) : null}
              </li>
            ))}
          </ul>
        </div>
        <div className={styles.narrativeBlock}>
          <h5>Recorded recommendations</h5>
          {recommendations.length ? (
            <ol className={styles.recommendations}>
              {recommendations.map((recommendation, index) => (
                <li key={`${index}-${recommendation}`}>{recommendation}</li>
              ))}
            </ol>
          ) : (
            <p>No recommendations were recorded.</p>
          )}
        </div>
      </div>
    </section>
  )
}

export function RunRecords({
  anchors,
  gateNames,
  runNames,
}: {
  anchors: EvaluationCampaignEvidenceAnchor[]
  gateNames: ReadonlyMap<string, string>
  runNames: ReadonlyMap<string, string>
}) {
  return (
    <section className={sectionStyles.technicalGroup} aria-labelledby="campaign-evidence-title">
      <div className={sectionStyles.technicalGroupHeader}>
        <div>
          <h4 id="campaign-evidence-title">Run records</h4>
          <p>Exact manifests, reports, and execution receipts used by this decision.</p>
        </div>
        <span>{anchors.length} verified</span>
      </div>
      <div className={styles.anchorGrid}>
        {anchors.map((anchor) => (
          <article
            key={`${anchor.slot_id}:${anchor.binding_role}`}
            className={styles.anchorCard}
            data-binding-role={anchor.binding_role}
            data-check-id={anchor.gate_id}
            data-slot-id={anchor.slot_id}
          >
            <div className={styles.anchorHeader}>
              <strong>{runNames.get(anchor.run_id) || anchor.run_id}</strong>
              <EvaluationTag>
                {gateNames.get(anchor.gate_id) || 'Release readiness'} ·{' '}
                {BINDING_ROLE_LABELS[anchor.binding_role]}
              </EvaluationTag>
            </div>
            <dl className={styles.digestList}>
              <div>
                <dt>Run ID</dt>
                <CopyableValue label="run ID" value={anchor.run_id} />
              </div>
              <div>
                <dt>Manifest identity</dt>
                <CopyableValue
                  label="manifest identity"
                  value={anchor.manifest_semantic_digest}
                  displayValue={shortIdentity(anchor.manifest_semantic_digest)}
                />
              </div>
              <div>
                <dt>Manifest artifact</dt>
                <CopyableValue
                  label="manifest artifact"
                  value={anchor.manifest_artifact_digest}
                  displayValue={shortIdentity(anchor.manifest_artifact_digest)}
                />
              </div>
              <div>
                <dt>Report receipt</dt>
                <CopyableValue
                  label="report receipt"
                  value={anchor.report_digest}
                  displayValue={shortIdentity(anchor.report_digest)}
                />
              </div>
              <div>
                <dt>Private execution receipt</dt>
                <CopyableValue
                  label="private execution receipt"
                  value={anchor.private_receipt_digest}
                  displayValue={shortIdentity(anchor.private_receipt_digest)}
                />
              </div>
              {anchor.candidate_subject_digest ? (
                <div>
                  <dt>Candidate configuration</dt>
                  <CopyableValue
                    label="candidate configuration"
                    value={anchor.candidate_subject_digest}
                    displayValue={shortIdentity(anchor.candidate_subject_digest)}
                  />
                </div>
              ) : null}
              {anchor.execution_attestation_digest ? (
                <div>
                  <dt>Server execution receipt</dt>
                  <CopyableValue
                    label="server execution receipt"
                    value={anchor.execution_attestation_digest}
                    displayValue={shortIdentity(anchor.execution_attestation_digest)}
                  />
                </div>
              ) : null}
            </dl>
          </article>
        ))}
      </div>
    </section>
  )
}

export function SupportingRecords({ gates }: { gates: EvaluationCampaignGate[] }) {
  const gatesWithRecords = gates.filter((gate) => gate.evidence_refs.length)
  if (!gatesWithRecords.length) return null

  return (
    <section
      className={sectionStyles.technicalGroup}
      aria-labelledby="campaign-supporting-records-title"
    >
      <div className={sectionStyles.technicalGroupHeader}>
        <h4 id="campaign-supporting-records-title">Supporting records</h4>
        <p>Exact service references associated with each release readiness check.</p>
      </div>
      <div className={styles.references}>
        {gatesWithRecords.map((gate) => (
          <article key={gate.id} data-check-id={gate.id}>
            <strong>{gate.name}</strong>
            <ul>
              {gate.evidence_refs.map((reference) => (
                <li key={reference} data-evidence-reference={reference}>
                  <span>{evidenceReferenceLabel(reference)}</span>
                  <code title={reference}>{reference}</code>
                </li>
              ))}
            </ul>
          </article>
        ))}
      </div>
    </section>
  )
}
