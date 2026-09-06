import type { EvaluationCampaign } from '../../types/evaluationCampaign'
import type { EvaluationRun } from '../../types/evaluationPlane'
import { EvaluationTechnicalDisclosure } from './EvaluationDisclosure'
import {
  ControlledComparisonDetails,
  FidelityDetails,
} from './EvaluationCampaignDecisionTechnicalComparisons'
import { TechnicalFieldGrid } from './EvaluationCampaignDecisionTechnicalFields'
import { copyField, digestField } from './evaluationTechnicalFields'
import {
  RecordedServiceNarrative,
  RunRecords,
  SupportingRecords,
} from './EvaluationCampaignDecisionTechnicalRecords'
import styles from './EvaluationCampaignDecisionTechnicalDetails.module.css'

interface EvaluationCampaignDecisionTechnicalDetailsProps {
  campaign: EvaluationCampaign
  runs: EvaluationRun[]
  createdAtLabel: string
}

export default function EvaluationCampaignDecisionTechnicalDetails({
  campaign,
  runs,
  createdAtLabel,
}: EvaluationCampaignDecisionTechnicalDetailsProps) {
  const decision = campaign.decision
  const gateNames = new Map(decision.gates.map((gate) => [gate.id, gate.name]))
  const runNames = new Map(runs.map((run) => [run.id, run.name]))

  return (
    <EvaluationTechnicalDisclosure
      className={styles.technicalDisclosure}
      summaryClassName={styles.technicalDisclosureSummary}
      summary={
        <>
          <span className={styles.technicalSummaryCopy}>
            <strong>Technical details</strong>
            <small>How this decision was verified and can be reproduced</small>
          </span>
          <span className={styles.technicalSummaryCount}>
            {decision.evidence.length} run {decision.evidence.length === 1 ? 'record' : 'records'}
          </span>
        </>
      }
    >
      <div className={styles.technicalBody}>
        <section
          className={styles.technicalGroup}
          aria-labelledby="campaign-technical-identity-title"
        >
          <div className={styles.technicalGroupHeader}>
            <h4 id="campaign-technical-identity-title">Decision identity</h4>
            <p>Immutable identifiers for verifying this release decision.</p>
          </div>
          <TechnicalFieldGrid
            label="Decision identity and receipts"
            fields={[
              copyField('Campaign ID', campaign.id),
              { label: 'Created', value: createdAtLabel },
              { label: 'Schema', value: decision.schema_version, mono: true },
              { label: 'Decision contract', value: decision.contract_version, mono: true },
              { label: 'Verification revision', value: decision.attestation_revision, mono: true },
              digestField('Evaluation receipt', campaign.manifest_digest),
              digestField('Decision receipt', decision.decision_digest),
            ]}
          />
        </section>

        <RecordedServiceNarrative
          summary={decision.summary}
          gates={decision.gates}
          recommendations={decision.recommendations}
        />
        {decision.paired_live_evidence ? (
          <ControlledComparisonDetails evidence={decision.paired_live_evidence} />
        ) : null}
        {decision.fidelity_evidence ? (
          <FidelityDetails evidence={decision.fidelity_evidence} />
        ) : null}
        <RunRecords anchors={decision.evidence} gateNames={gateNames} runNames={runNames} />
        <SupportingRecords gates={decision.gates} />
      </div>
    </EvaluationTechnicalDisclosure>
  )
}
