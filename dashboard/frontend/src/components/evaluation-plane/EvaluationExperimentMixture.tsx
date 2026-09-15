import type { EvaluationCatalogTarget } from '../../types/evaluationPlane'
import EvaluationDisclosure, { EvaluationTechnicalDisclosure } from './EvaluationDisclosure'
import noticeStyles from './EvaluationExperimentNotice.module.css'
import styles from './EvaluationExperimentMixture.module.css'
import type { EvaluationExperimentFormModel } from './useEvaluationExperimentForm'

type EvaluationMixture = NonNullable<EvaluationCatalogTarget['mixture']>

function shortDigest(value: string): string {
  return `${value.slice(0, 14)}…${value.slice(-8)}`
}

function MixtureHeader({ mixture }: { mixture: EvaluationMixture }) {
  return (
    <>
      <span className={styles.mixtureSummaryCopy}>
        <span className={styles.mixtureLabel}>Selected Mixture-of-Models</span>
        <strong>{mixture.entrypoint_model}</strong>
        <small>
          Recipe <code>{mixture.recipe_name}</code> and its reachable model pool will be saved with
          this evaluation run.
        </small>
      </span>
      <span className={styles.mixtureFacts}>
        <span>
          {mixture.model_arms.length} pool {mixture.model_arms.length === 1 ? 'model' : 'models'}
        </span>
        <span>
          {mixture.decisions.length} {mixture.decisions.length === 1 ? 'decision' : 'decisions'}
        </span>
        <span>
          {mixture.aliases.length} public {mixture.aliases.length === 1 ? 'name' : 'names'}
        </span>
      </span>
    </>
  )
}

function MixtureDecisions({ mixture }: { mixture: EvaluationMixture }) {
  const armsByID = new Map(mixture.model_arms.map((arm) => [arm.id, arm]))
  return (
    <div>
      <span className={styles.mixtureLabel}>Recipe decisions</span>
      <div className={styles.decisionList}>
        {mixture.decisions.map((decision) => (
          <article key={decision.name}>
            <div>
              <strong>{decision.name}</strong>
              <span>Routing strategy configured</span>
              <small>
                {decision.arm_ids
                  .map((armID) => armsByID.get(armID)?.model || 'Unresolved pool model')
                  .join(' · ')}
              </small>
            </div>
            <span>
              {decision.arm_ids.length} eligible{' '}
              {decision.arm_ids.length === 1 ? 'model' : 'models'}
            </span>
          </article>
        ))}
      </div>
    </div>
  )
}

function MixtureModelPool({ mixture }: { mixture: EvaluationMixture }) {
  return (
    <div>
      <span className={styles.mixtureLabel}>Model pool</span>
      <div className={styles.armList}>
        {mixture.model_arms.map((arm) => (
          <article key={arm.id}>
            <strong>{arm.model}</strong>
            <span>
              {(arm.modalities || ['text']).join(' · ')}
              {arm.parameter_size ? ` · ${arm.parameter_size}` : ''}
            </span>
            <small>
              ${arm.input_cost_per_million_tokens_usd.toLocaleString()}/M in · $
              {arm.output_cost_per_million_tokens_usd.toLocaleString()}/M out
            </small>
          </article>
        ))}
      </div>
      {mixture.support_models.length ? (
        <p className={styles.supportModels}>
          Decision support only (not scored as pool models):{' '}
          {mixture.support_models.map((model) => model.model).join(' · ')}
        </p>
      ) : null}
    </div>
  )
}

function MixtureLineage({ mixture }: { mixture: EvaluationMixture }) {
  const poolArmIDs = new Set(mixture.model_arms.map((arm) => arm.id))
  const unresolvedArmIDs = [
    ...new Set(
      mixture.decisions.flatMap((decision) =>
        decision.arm_ids.filter((armID) => !poolArmIDs.has(armID)),
      ),
    ),
  ]
  return (
    <EvaluationTechnicalDisclosure
      className={styles.mixtureLineageDetails}
      summary="Reproducibility details"
      summaryClassName={styles.mixtureLineageSummary}
    >
      <footer className={styles.mixtureLineage}>
        <span>
          Recipe <code title={mixture.recipe_digest}>{shortDigest(mixture.recipe_digest)}</code>
        </span>
        <span>
          Pool <code title={mixture.pool_digest}>{shortDigest(mixture.pool_digest)}</code>
        </span>
        <span>
          Selector{' '}
          <code title={mixture.selector_digest}>{shortDigest(mixture.selector_digest)}</code>
        </span>
        <span>
          Adaptation{' '}
          <code title={mixture.adaptation_digest}>{shortDigest(mixture.adaptation_digest)}</code>
        </span>
        <span>
          Binding <code title={mixture.binding_digest}>{shortDigest(mixture.binding_digest)}</code>
        </span>
        <span>
          Routing methods{' '}
          <code>
            {mixture.decisions
              .map((decision) => `${decision.name}: ${decision.algorithm}`)
              .join(' · ')}
          </code>
        </span>
        {unresolvedArmIDs.length ? (
          <span>
            Unresolved pool model IDs <code>{unresolvedArmIDs.join(' · ')}</code>
          </span>
        ) : null}
      </footer>
    </EvaluationTechnicalDisclosure>
  )
}

export default function EvaluationExperimentMixture({
  target,
  form,
}: {
  target: EvaluationCatalogTarget | undefined
  form: EvaluationExperimentFormModel
}) {
  if (form.mode !== 'live') return null
  const mixture = target?.mixture
  if (!mixture) {
    return (
      <div className={`${noticeStyles.contractWarning} ${styles.fieldWide}`} role="status">
        Select an available Mixture-of-Models configuration before starting a live evaluation.
      </div>
    )
  }
  return (
    <>
      {target.healthy === false ? (
        <div className={`${noticeStyles.contractWarning} ${styles.fieldWide}`} role="alert">
          This Mixture is visible but not ready. Connect its models and evaluation runtime before
          starting a live evaluation.
        </div>
      ) : null}
      <EvaluationDisclosure
        className={`${styles.mixtureSnapshot} ${styles.fieldWide}`}
        aria-label="Selected Mixture-of-Models details"
        indicator="label"
        summary={<MixtureHeader mixture={mixture} />}
        summaryClassName={styles.mixtureHeader}
      >
        <div className={styles.mixtureColumns}>
          <MixtureDecisions mixture={mixture} />
          <MixtureModelPool mixture={mixture} />
        </div>
        <MixtureLineage mixture={mixture} />
      </EvaluationDisclosure>
    </>
  )
}
