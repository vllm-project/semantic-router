import type { EvaluationChangeProfileId, EvaluationCatalog } from '../../types/evaluationPlane'
import type { EvaluationExperimentFormModel } from './useEvaluationExperimentForm'
import EvaluationExperimentSectionHeading from './EvaluationExperimentSectionHeading'
import EvaluationDisclosure from './EvaluationDisclosure'
import { EvaluationTag } from './EvaluationPrimitives'
import styles from './EvaluationExperimentGateScope.module.css'
import noticeStyles from './EvaluationExperimentNotice.module.css'
import sectionStyles from './EvaluationExperimentSection.module.css'

interface EvaluationExperimentGateScopeProps {
  catalog: EvaluationCatalog
  form: EvaluationExperimentFormModel
}

export default function EvaluationExperimentGateScope({
  catalog,
  form,
}: EvaluationExperimentGateScopeProps) {
  const requiredGates = form.gateApplicability.filter(
    (gate) => gate.disposition === 'required',
  ).length
  const advisoryGates = form.gateApplicability.filter(
    (gate) => gate.disposition === 'advisory',
  ).length
  return (
    <section className={sectionStyles.formSection}>
      <EvaluationExperimentSectionHeading
        index="02"
        title="Release readiness"
        description="The change type selects the checks required before this result can support a production decision."
      />
      <div className={styles.profileHeader}>
        <label>
          Change type
          <select
            value={form.changeProfile}
            disabled={form.baselineLocked}
            onChange={(event) =>
              form.setChangeProfile(event.target.value as EvaluationChangeProfileId)
            }
            required
          >
            <option value="">Select change type</option>
            {catalog.change_profiles.map((profile) => (
              <option key={profile.id} value={profile.id}>
                {profile.name}
              </option>
            ))}
          </select>
          <small>
            {form.selectedChangeProfile?.description ||
              'Only registered change types are selectable.'}
          </small>
        </label>
        <div>
          <span>Release checks</span>
          <strong>
            {requiredGates} required · {advisoryGates} recommended
          </strong>
        </div>
      </div>
      {form.gateApplicability.length ? (
        <EvaluationDisclosure
          className={styles.gateDisclosure}
          summaryClassName={styles.gateDisclosureSummary}
          summary={
            <>
              <span>Review release checks</span>
              <small>
                {requiredGates} required · {advisoryGates} recommended ·{' '}
                {form.gateApplicability.length - requiredGates - advisoryGates} not required
              </small>
            </>
          }
        >
          <div className={styles.gateMatrix} role="list" aria-label="Release check applicability">
            {form.gateApplicability.map((gate) => (
              <article key={gate.gate_id} role="listitem" data-disposition={gate.disposition}>
                <div>
                  <strong>{gate.name}</strong>
                </div>
                <EvaluationTag
                  tone={
                    gate.disposition === 'required'
                      ? 'positive'
                      : gate.disposition === 'advisory'
                        ? 'warning'
                        : 'neutral'
                  }
                >
                  {gate.disposition === 'required'
                    ? 'Required'
                    : gate.disposition === 'advisory'
                      ? 'Recommended'
                      : 'Not required'}
                </EvaluationTag>
                <small>{gate.description}</small>
              </article>
            ))}
          </div>
        </EvaluationDisclosure>
      ) : (
        <div className={noticeStyles.contractWarning} role="status">
          Release checks for this evaluation catalog could not be explained. The completed report
          remains the source of truth for a production decision.
        </div>
      )}
    </section>
  )
}
