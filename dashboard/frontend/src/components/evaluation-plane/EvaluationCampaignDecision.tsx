import { useEffect, useRef } from 'react'

import type { EvaluationCampaign } from '../../types/evaluationCampaign'
import type { EvaluationRun } from '../../types/evaluationPlane'
import {
  formatCampaignCreatedAt,
  releaseNextActions,
  requiredCheckCounts,
} from './evaluationCampaignDecisionPresentation'
import {
  EvaluationCampaignDecisionSummary,
  EvaluationCampaignFidelity,
  EvaluationCampaignGateResults,
  EvaluationCampaignNextActions,
} from './EvaluationCampaignDecisionSections'
import EvaluationCampaignDecisionTechnicalDetails from './EvaluationCampaignDecisionTechnicalDetails'
import EvaluationCampaignPairedOutcomes from './EvaluationCampaignPairedOutcomes'
import styles from './EvaluationCampaignDecisionLayout.module.css'

interface EvaluationCampaignDecisionProps {
  campaign: EvaluationCampaign
  runs: EvaluationRun[]
  onStartAnother: () => void
}

export default function EvaluationCampaignDecision({
  campaign,
  runs,
  onStartAnother,
}: EvaluationCampaignDecisionProps) {
  const titleRef = useRef<HTMLHeadingElement>(null)
  const counts = requiredCheckCounts(campaign)
  const actions = releaseNextActions(campaign.decision.verdict, counts)
  useEffect(() => {
    titleRef.current?.focus()
  }, [campaign.id])

  return (
    <article className={styles.decision} aria-labelledby="evaluation-campaign-decision-title">
      <EvaluationCampaignDecisionSummary
        campaign={campaign}
        counts={counts}
        titleRef={titleRef}
        onStartAnother={onStartAnother}
      />
      {campaign.decision.paired_live_evidence ? (
        <EvaluationCampaignPairedOutcomes
          evidence={campaign.decision.paired_live_evidence}
          runs={runs}
        />
      ) : null}
      <EvaluationCampaignFidelity campaign={campaign} />
      <EvaluationCampaignGateResults campaign={campaign} />
      <EvaluationCampaignNextActions createdAt={campaign.decision.created_at} actions={actions} />
      <EvaluationCampaignDecisionTechnicalDetails
        campaign={campaign}
        runs={runs}
        createdAtLabel={formatCampaignCreatedAt(campaign.decision.created_at)}
      />
    </article>
  )
}
