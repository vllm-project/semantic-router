import { buildEvaluationCampaignRequest } from './evaluationCampaignSupport'
import {
  EvaluationCampaignBuilderFooter,
  EvaluationCampaignBuilderHeader,
  EvaluationCampaignEvidenceInputs,
  EvaluationCampaignIdentityFields,
} from './EvaluationCampaignBuilderSections'
import type { EvaluationCampaignBuilderProps } from './evaluationCampaignBuilderTypes'
import useEvaluationCampaignBuilderViewModel from './useEvaluationCampaignBuilderViewModel'
import styles from './EvaluationCampaignBuilder.module.css'

export default function EvaluationCampaignBuilder(builder: EvaluationCampaignBuilderProps) {
  const view = useEvaluationCampaignBuilderViewModel(builder)
  return (
    <form
      className={styles.builder}
      aria-busy={builder.createPending}
      onSubmit={(event) => {
        event.preventDefault()
        if (builder.model.validation || !builder.canCreate || builder.createPending) return
        void builder.onCreate(buildEvaluationCampaignRequest(builder.model.draft))
      }}
    >
      <EvaluationCampaignBuilderHeader builder={builder} view={view} />
      <EvaluationCampaignEvidenceInputs builder={builder} view={view} />
      <EvaluationCampaignIdentityFields builder={builder} />
      <EvaluationCampaignBuilderFooter builder={builder} />
    </form>
  )
}
