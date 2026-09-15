import type { RefObject } from 'react'

import type { EvaluationCampaign } from '../../types/evaluationCampaign'
import {
  formatCampaignCreatedAt,
  formatCampaignStatistic,
  formatCampaignThreshold,
  gateSourceLabel,
  releaseDecisionSummary,
  type RequiredCheckCounts,
} from './evaluationCampaignDecisionPresentation'
import { EvaluationActionButton, GateVerdictBadge } from './EvaluationPrimitives'
import {
  evaluationResultScopeLabel,
  formatMetric,
  gateVerdictPresentation,
} from './evaluationPresentation'
import { changeProfileLabel } from './evaluationRunPresentation'
import commonStyles from './EvaluationCampaign.module.css'
import styles from './EvaluationCampaignDecisionLayout.module.css'
import fidelityStyles from './EvaluationCampaignFidelityEvidence.module.css'
import gateStyles from './EvaluationGateList.module.css'

export function EvaluationCampaignDecisionSummary({
  campaign,
  counts,
  titleRef,
  onStartAnother,
}: {
  campaign: EvaluationCampaign
  counts: RequiredCheckCounts
  titleRef: RefObject<HTMLHeadingElement>
  onStartAnother: () => void
}) {
  return (
    <>
      <header className={styles.decisionHero}>
        <div>
          <span className={commonStyles.eyebrow}>Verified release decision</span>
          <h3 id="evaluation-campaign-decision-title" ref={titleRef} tabIndex={-1}>
            {campaign.name}
          </h3>
          <p>{releaseDecisionSummary(campaign.decision.verdict, counts)}</p>
        </div>
        <div className={styles.decisionHeroActions}>
          <GateVerdictBadge verdict={campaign.decision.verdict} disposition="required" />
          <EvaluationActionButton type="button" onClick={onStartAnother}>
            Start another decision
          </EvaluationActionButton>
        </div>
      </header>
      <dl className={styles.decisionMeta} aria-label="Release decision summary">
        <div>
          <dt>Required checks</dt>
          <dd>
            {counts.passed} passed · {counts.failed} blocked · {counts.unavailable} incomplete
          </dd>
        </div>
        <div>
          <dt>Change type</dt>
          <dd>{changeProfileLabel(campaign.change_profile)}</dd>
        </div>
      </dl>
    </>
  )
}

export function EvaluationCampaignFidelity({ campaign }: { campaign: EvaluationCampaign }) {
  const evidence = campaign.decision.fidelity_evidence
  if (!evidence) return null
  const measures = [
    ['Matched', evidence.matched_cases],
    ['Decision drift', evidence.decision_mismatches],
    ['Outcome drift', evidence.outcome_mismatches],
    ['Not measured', evidence.unavailable_cases],
    ['Agreement', formatCampaignStatistic(evidence.point_estimate)],
    ['One-sided lower bound', formatCampaignStatistic(evidence.lower_bound)],
  ]
  return (
    <section className={styles.decisionSection} aria-labelledby="campaign-fidelity-title">
      <div className={styles.sectionHeader}>
        <div>
          <span className={commonStyles.eyebrow}>Reference → fresh live</span>
          <h4 id="campaign-fidelity-title">Live consistency</h4>
          <p>
            Fresh live outcomes are compared with the reference result over the same cases to detect
            decision or outcome drift.
          </p>
        </div>
      </div>
      <div className={fidelityStyles.fidelitySummary}>
        <dl className={fidelityStyles.measures} aria-label="Live fidelity decision statistics">
          {measures.map(([label, value]) => (
            <div key={label}>
              <dt>{label}</dt>
              <dd>{value}</dd>
            </div>
          ))}
          <div className={fidelityStyles.verdict} data-check-id="G5">
            <dt>Live fidelity result</dt>
            <dd>
              <GateVerdictBadge verdict={evidence.verdict} disposition="required" />
            </dd>
          </div>
        </dl>
      </div>
    </section>
  )
}

function EvaluationCampaignGateRow({
  gate,
}: {
  gate: EvaluationCampaign['decision']['gates'][number]
}) {
  const presentation = gateVerdictPresentation(gate)
  return (
    <article
      className={gateStyles.gateRow}
      data-check-id={gate.id}
      data-evidence-level={gate.evidence_level}
      data-source-id={gate.source}
      data-verdict={gate.verdict}
      data-tone={presentation.tone}
    >
      <div className={gateStyles.gateIdentity}>
        <div>
          <strong>{gate.name}</strong>
        </div>
        <p>{presentation.explanation}</p>
      </div>
      <div className={gateStyles.gateEvidence}>
        <span className={gateStyles.gateEvidenceMeta}>
          {gateSourceLabel(gate.source)} · {evaluationResultScopeLabel(gate.evidence_level)}
        </span>
        <span>
          {gate.observed === undefined
            ? 'No numeric result was recorded'
            : `Measured ${formatMetric({ value: gate.observed, unit: gate.threshold?.unit || '' })}`}
          {gate.threshold ? ` · target ${formatCampaignThreshold(gate.threshold)}` : ''}
          {gate.sample_count === undefined
            ? ''
            : ` · ${gate.sample_count} ${gate.sample_count === 1 ? 'sample' : 'samples'}`}
        </span>
      </div>
      <span className={gateStyles.gateVerdict}>
        <GateVerdictBadge verdict={gate.verdict} disposition={gate.disposition} />
      </span>
    </article>
  )
}

export function EvaluationCampaignGateResults({ campaign }: { campaign: EvaluationCampaign }) {
  const gateCount = campaign.decision.gates.length
  return (
    <section className={styles.decisionSection} aria-labelledby="campaign-gates-title">
      <div className={styles.sectionHeader}>
        <div>
          <span className={commonStyles.eyebrow}>Release boundary</span>
          <h4 id="campaign-gates-title">Release readiness checks</h4>
          <p>See which required and recommended checks passed, were blocked, or need results.</p>
        </div>
        <span className={styles.sectionMeta}>
          {gateCount} {gateCount === 1 ? 'check' : 'checks'}
        </span>
      </div>
      <div className={gateStyles.gateList}>
        {campaign.decision.gates.map((gate) => (
          <EvaluationCampaignGateRow key={gate.id} gate={gate} />
        ))}
      </div>
    </section>
  )
}

export function EvaluationCampaignNextActions({
  createdAt,
  actions,
}: {
  createdAt: string
  actions: string[]
}) {
  return (
    <section className={styles.decisionSection} aria-labelledby="campaign-actions-title">
      <div className={styles.sectionHeader}>
        <div>
          <span className={commonStyles.eyebrow}>Next actions</span>
          <h4 id="campaign-actions-title">What to do next</h4>
        </div>
        <time className={styles.sectionMeta} dateTime={createdAt}>
          {formatCampaignCreatedAt(createdAt)}
        </time>
      </div>
      <ol className={gateStyles.nextActions}>
        {actions.map((action) => (
          <li key={action}>{action}</li>
        ))}
      </ol>
    </section>
  )
}
