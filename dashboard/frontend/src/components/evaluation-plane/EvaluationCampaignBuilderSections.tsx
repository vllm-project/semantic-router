import EvaluationCampaignControlledPair from './EvaluationCampaignControlledPair'
import EvaluationCampaignSlotBinding from './EvaluationCampaignSlotBinding'
import EvaluationDisclosure from './EvaluationDisclosure'
import type { EvaluationCampaignBuilderProps } from './evaluationCampaignBuilderTypes'
import type { EvaluationCampaignBuilderViewModel } from './useEvaluationCampaignBuilderViewModel'
import EvaluationIssueDetails from './EvaluationIssueDetails'
import { EvaluationActionButton, EvaluationTag } from './EvaluationPrimitives'
import { campaignSlotRunIDs } from './evaluationCampaignSupport'
import commonStyles from './EvaluationCampaign.module.css'
import matrixStyles from './EvaluationCampaignBuilderMatrix.module.css'
import sectionStyles from './EvaluationCampaignBuilderSections.module.css'

type SectionProps = {
  builder: EvaluationCampaignBuilderProps
  view: EvaluationCampaignBuilderViewModel
}

function slotDisposition(value: string): string {
  if (value === 'required') return 'Required'
  if (value === 'advisory') return 'Optional'
  return 'Not applicable'
}

function slotSelectionStatus(selected: boolean, disposition: string): string {
  if (selected) return 'Run selected'
  if (disposition === 'required') return 'Run required'
  if (disposition === 'advisory') return 'May be omitted'
  return 'No run needed'
}

export function EvaluationCampaignBuilderHeader({ builder, view }: SectionProps) {
  const { model } = builder
  return (
    <>
      <div className={sectionStyles.builderHeader}>
        <span className={commonStyles.eyebrow}>Release readiness</span>
        <EvaluationTag>{builder.totalRuns} runs available</EvaluationTag>
      </div>
      {!builder.allRunsLoaded ? (
        <div className={sectionStyles.ledgerBoundary} role="status">
          <div>
            <strong>Load all runs to continue</strong>
            <span>
              {builder.runs.length} of {builder.totalRuns} runs are loaded. The decision stays
              locked until every eligible result can be considered.
            </span>
          </div>
          <EvaluationActionButton
            type="button"
            disabled={
              builder.createPending || builder.loadingAllRuns || !builder.runLedgerAvailable
            }
            onClick={builder.onLoadAllRuns}
          >
            {builder.loadingAllRuns ? 'Loading all runs…' : 'Load all runs'}
          </EvaluationActionButton>
        </div>
      ) : null}
      <div className={sectionStyles.profileGrid}>
        <label className={sectionStyles.field}>
          <span>Change type</span>
          <select
            aria-label="Release decision change type"
            value={model.draft.changeProfile}
            disabled={builder.createPending || view.profileLocked}
            onChange={(event) =>
              model.changeProfile(event.target.value as typeof model.draft.changeProfile)
            }
          >
            {builder.catalog.change_profiles.map((profile) => (
              <option key={profile.id} value={profile.id}>
                {profile.name}
              </option>
            ))}
          </select>
          <small>
            {view.profileLocked
              ? 'The change type is locked while this controlled comparison is active.'
              : model.profile?.description || 'Checks selected for this type of change'}
          </small>
        </label>
        <dl className={sectionStyles.profileSummary} aria-label="Release readiness summary">
          <div>
            <dt>Required checks</dt>
            <dd>
              {view.readyRequiredSlots} / {model.requiredSlotCount} bound
            </dd>
          </div>
          <div>
            <dt>Optional checks</dt>
            <dd>{model.advisorySlotCount} optional</dd>
          </div>
          <div>
            <dt>A/B validation</dt>
            <dd>
              {view.g3?.disposition === 'not_applicable'
                ? 'Not required'
                : 'Fresh comparison required'}
            </dd>
          </div>
        </dl>
      </div>
    </>
  )
}

function EvaluationCampaignControlledInput({ builder, view }: SectionProps) {
  if (
    !view.g3 ||
    view.g3.disposition === 'not_applicable' ||
    !builder.model.profile
  ) {
    return null
  }
  return (
    <EvaluationCampaignControlledPair
      key={builder.model.draft.changeProfile}
      runs={builder.runs}
      profile={builder.model.profile}
      slot={view.g3}
      readiness={builder.model.readiness}
      canCreate={builder.canCreate}
      disabled={view.inputDisabled}
      activePairID={builder.activeControlledPairID}
      resumablePair={view.resumablePair}
      onProfileLockChange={view.setControlledPairProfileLocked}
      onPairIdentityChange={builder.onControlledPairIdentityChange}
      onReady={view.onControlledPairReady}
    />
  )
}

function EvaluationCampaignRoleMatrix({ builder, view }: SectionProps) {
  return (
    <div
      className={matrixStyles.roleMatrixFrame}
      role="region"
      aria-label="Release decision inputs"
    >
      <table className={matrixStyles.roleMatrix}>
        <thead>
          <tr>
            <th scope="col">Release check</th>
            <th scope="col">Status</th>
            <th scope="col">Selected runs</th>
          </tr>
        </thead>
        <tbody>
          {builder.model.slots.map((slot) => {
            const ids = campaignSlotRunIDs(slot, builder.model.draft.gateBindings)
            const selected = ids.length === (slot.binding_kind === 'run' ? 1 : 2)
            const required = slot.disposition === 'required'
            return (
              <tr key={slot.gate_id} data-gate={slot.gate_id} data-selected={selected}>
                <th scope="row">
                  <strong>{slot.name}</strong>
                  <small>{slot.description}</small>
                </th>
                <td data-label="Status">
                  <EvaluationTag tone={required ? 'positive' : 'neutral'}>
                    {slotDisposition(slot.disposition)}
                  </EvaluationTag>
                  <small className={selected ? matrixStyles.roleReady : matrixStyles.roleEmpty}>
                    {slotSelectionStatus(selected, slot.disposition)}
                  </small>
                </td>
                <td data-label="Selected runs">
                  <EvaluationCampaignSlotBinding
                    slot={slot}
                    runs={builder.runs}
                    model={builder.model}
                    disabled={view.inputDisabled}
                  />
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}

export function EvaluationCampaignEvidenceInputs({ builder, view }: SectionProps) {
  const remaining = view.requiredSlots.length - view.readyRequiredSlots
  return (
    <EvaluationDisclosure
      className={matrixStyles.evidenceDisclosure}
      focus="outside"
      summaryClassName={matrixStyles.evidenceDisclosureSummary}
      summary={
        <>
          <span>
            <strong>Review evaluation inputs</strong>
            <small>
              {remaining === 0
                ? 'Every required check has a run selected.'
                : `${remaining} required ${remaining === 1 ? 'check still needs' : 'checks still need'} a run.`}
            </small>
          </span>
          <b>
            {view.readyRequiredSlots} of {view.requiredSlots.length} ready
          </b>
        </>
      }
    >
      <div className={matrixStyles.evidenceDisclosureBody}>
        <EvaluationCampaignControlledInput builder={builder} view={view} />
        <EvaluationCampaignRoleMatrix builder={builder} view={view} />
      </div>
    </EvaluationDisclosure>
  )
}

export function EvaluationCampaignIdentityFields({
  builder,
}: {
  builder: EvaluationCampaignBuilderProps
}) {
  const { model } = builder
  return (
    <div className={sectionStyles.formGrid}>
      <label className={sectionStyles.field}>
        <span>Decision name</span>
        <input
          aria-label="Decision name"
          value={model.draft.name}
          maxLength={200}
          disabled={builder.createPending}
          placeholder="e.g. Recipe v4 production review"
          onChange={(event) =>
            model.revise((current) => ({ ...current, name: event.target.value }))
          }
        />
      </label>
      <EvaluationDisclosure
        className={sectionStyles.contextDisclosure}
        summaryClassName={sectionStyles.contextDisclosureSummary}
        summary={
          <>
            <span>Decision notes</span>
            <small>{model.draft.description ? 'Notes added' : 'Optional context'}</small>
          </>
        }
      >
        <label className={`${sectionStyles.field} ${sectionStyles.contextField}`}>
          <span className={commonStyles.srOnly}>Decision notes</span>
          <textarea
            aria-label="Decision notes"
            value={model.draft.description}
            maxLength={4000}
            disabled={builder.createPending}
            placeholder="What changed, who is affected, and what decision needs to be made."
            onChange={(event) =>
              model.revise((current) => ({ ...current, description: event.target.value }))
            }
          />
        </label>
      </EvaluationDisclosure>
    </div>
  )
}

export function EvaluationCampaignBuilderFooter({
  builder,
}: {
  builder: EvaluationCampaignBuilderProps
}) {
  return (
    <>
      {builder.createError ? (
        <div
          className={`${commonStyles.inlineError} ${sectionStyles.builderNotice}`}
          role="alert"
        >
          <div>
            <strong>Release decision could not be created</strong>
            <span>Review the selected evidence, then retry.</span>
            <EvaluationIssueDetails
              issues={[{ label: 'Create decision request', message: builder.createError }]}
            />
          </div>
          <EvaluationActionButton
            type="button"
            compact
            disabled={builder.createPending}
            onClick={builder.onClearCreateError}
          >
            Dismiss
          </EvaluationActionButton>
        </div>
      ) : null}
      {!builder.canCreate ? (
        <div
          className={`${commonStyles.inlineNotice} ${sectionStyles.builderNotice}`}
          role="status"
        >
          <div>
            <strong>Read-only decision workspace</strong>
            <span>Evaluation write permission is required to publish a release decision.</span>
          </div>
        </div>
      ) : null}
      <div className={sectionStyles.formActions}>
        <span>
          {builder.model.validation ||
            'All required checks are ready. Selected runs will be verified.'}
        </span>
        <EvaluationActionButton
          type="submit"
          variant="primary"
          disabled={
            Boolean(builder.model.validation) || !builder.canCreate || builder.createPending
          }
        >
          {builder.createPending ? 'Creating decision…' : 'Create release decision'}
        </EvaluationActionButton>
      </div>
    </>
  )
}
