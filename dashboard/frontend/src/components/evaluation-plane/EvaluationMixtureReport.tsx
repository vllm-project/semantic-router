import type { EvaluationMetric, EvaluationReport } from '../../types/evaluationReport'
import { EvaluationTechnicalDisclosure } from './EvaluationDisclosure'
import EvaluationIssueDetails from './EvaluationIssueDetails'
import { formatMetric } from './evaluationPresentation'
import { EvaluationTag } from './EvaluationPrimitives'
import styles from './EvaluationMixtureReport.module.css'
import layoutStyles from './EvaluationReportLayout.module.css'

type EvaluationMixture = NonNullable<EvaluationReport['run']['mixture']>
type EvaluationModelArm = EvaluationMixture['model_arms'][number]

function metricByID(metrics: EvaluationMetric[], id: string): EvaluationMetric | undefined {
  return metrics.find((metric) => metric.id === id)
}

function MetricReading({ metric }: { metric: EvaluationMetric | undefined }) {
  return metric?.value !== null && typeof metric?.value === 'number' ? (
    <strong>{formatMetric(metric)}</strong>
  ) : (
    <strong className={styles.mixtureMissing}>Not measured</strong>
  )
}

function OutcomeLayer({
  eyebrow,
  title,
  description,
  readings,
}: {
  eyebrow: string
  title: string
  description: string
  readings: Array<{ label: string; metric: EvaluationMetric | undefined }>
}) {
  return (
    <article className={styles.mixtureOutcomeLayer}>
      <span>{eyebrow}</span>
      <h4>{title}</h4>
      <p>{description}</p>
      <dl>
        {readings.map((reading) => (
          <div key={reading.label}>
            <dt>{reading.label}</dt>
            <dd>
              <MetricReading metric={reading.metric} />
            </dd>
          </div>
        ))}
      </dl>
    </article>
  )
}

function decisionNamesByArm(mixture: EvaluationMixture): Map<string, string[]> {
  const decisionsByArm = new Map<string, string[]>()
  mixture.decisions.forEach((decision, index) => {
    for (const armID of decision.arm_ids) {
      decisionsByArm.set(armID, [...(decisionsByArm.get(armID) || []), `Decision ${index + 1}`])
    }
  })
  return decisionsByArm
}

function MixtureHeader({ mixture }: { mixture: EvaluationMixture }) {
  return (
    <div className={layoutStyles.sectionHeader}>
      <div>
        <span className={layoutStyles.eyebrow}>Evaluated system boundary</span>
        <h3 id="mixture-report-title">{mixture.entrypoint_model}</h3>
        <p>
          This run measured recipe decisions, every reachable model, and the routed system outcome.
          The setup below is the one that actually ran—not the current configuration.
        </p>
      </div>
      <div className={styles.mixtureSubjectFacts}>
        <EvaluationTag>
          {mixture.model_arms.length} {mixture.model_arms.length === 1 ? 'model' : 'models'}
        </EvaluationTag>
        <EvaluationTag>
          {mixture.decisions.length} {mixture.decisions.length === 1 ? 'decision' : 'decisions'}
        </EvaluationTag>
        <EvaluationTag>
          {mixture.aliases.length} entrypoint {mixture.aliases.length === 1 ? 'name' : 'names'}
        </EvaluationTag>
      </div>
    </div>
  )
}

function MixtureOutcomes({
  metrics,
  mixture,
}: {
  metrics: EvaluationMetric[]
  mixture: EvaluationMixture
}) {
  return (
    <>
      <div className={styles.mixtureOutcomeGrid}>
        <OutcomeLayer
          eyebrow="01 · Routing recipe"
          title={mixture.recipe_name}
          description="Does the recipe select an eligible model for the right reason?"
          readings={[
            { label: 'Decision accuracy', metric: metricByID(metrics, 'routing.accuracy') },
            { label: 'Coverage', metric: metricByID(metrics, 'routing.coverage') },
            { label: 'Fallback rate', metric: metricByID(metrics, 'routing.fallback_rate') },
          ]}
        />
        <OutcomeLayer
          eyebrow="02 · Model pool"
          title="Capability frontier"
          description="How good and complementary are the saved models before routing?"
          readings={[
            {
              label: 'Best available model quality',
              metric: metricByID(metrics, 'model_pool.oracle_quality'),
            },
            {
              label: 'Best single model',
              metric: metricByID(metrics, 'model_pool.best_single_quality'),
            },
            {
              label: 'Gain over the best single model',
              metric: metricByID(metrics, 'model_pool.oracle_gain'),
            },
          ]}
        />
        <OutcomeLayer
          eyebrow="03 · Routed system"
          title="Realized utility"
          description="How much of the pool frontier does the recipe capture in practice?"
          readings={[
            { label: 'Realized quality', metric: metricByID(metrics, 'joint.realized_quality') },
            {
              label: 'Normalized quality gap',
              metric: metricByID(metrics, 'joint.normalized_regret'),
            },
            {
              label: 'Share of best-available quality delivered',
              metric: metricByID(metrics, 'joint.oracle_capture_ratio'),
            },
          ]}
        />
      </div>
      <p className={styles.mixtureReadingGuide}>
        Read left to right: the recipe chooses, the model comparison establishes the pool ceiling,
        and the routed call measures how much of that ceiling the system realizes. “Not measured”
        means the selected test setup did not produce that aggregate; the dashboard never
        substitutes a different target.
      </p>
    </>
  )
}

function MixtureDecisionTopology({ mixture }: { mixture: EvaluationMixture }) {
  const armsByID = new Map(mixture.model_arms.map((arm) => [arm.id, arm]))
  return (
    <div>
      <div className={styles.mixtureSubheading}>
        <div>
          <span>Recipe topology</span>
          <strong>Decision → eligible models</strong>
        </div>
      </div>
      <div className={styles.mixtureDecisionMap}>
        {mixture.decisions.map((decision, index) => (
          <MixtureDecision
            key={decision.name}
            label={`Decision ${index + 1}`}
            decision={decision}
            armsByID={armsByID}
          />
        ))}
      </div>
      {mixture.support_models.length ? (
        <p className={styles.mixtureSupportModels}>
          <strong>Decision support models (not evaluated pool models)</strong>
          <span>{mixture.support_models.map((model) => model.model).join(' · ')}</span>
        </p>
      ) : null}
    </div>
  )
}

function MixtureDecision({
  label,
  decision,
  armsByID,
}: {
  label: string
  decision: EvaluationMixture['decisions'][number]
  armsByID: Map<string, EvaluationModelArm>
}) {
  const unresolvedArmIDs = decision.arm_ids.filter((armID) => !armsByID.has(armID))
  return (
    <article>
      <div>
        <strong>{label}</strong>
        <span>Routing strategy configured</span>
      </div>
      <span>
        {decision.arm_ids
          .map((armID) => armsByID.get(armID)?.model || 'Unresolved pool model')
          .join(' · ')}
      </span>
      <EvaluationIssueDetails
        issues={unresolvedArmIDs.map((armID) => ({
          label: 'Unresolved model reference',
          message: armID,
        }))}
      />
    </article>
  )
}

function MixtureArmCard({
  arm,
  metrics,
  mixture,
  decisions,
}: {
  arm: EvaluationModelArm
  metrics: EvaluationMetric[]
  mixture: EvaluationMixture
  decisions: string[]
}) {
  const quality = metricByID(metrics, `model_pool.arm.${arm.id}.quality`)
  const success = metricByID(metrics, `model_pool.arm.${arm.id}.success_rate`)
  const contribution = metricByID(metrics, `model_pool.arm.${arm.id}.marginal_contribution`)
  return (
    <article>
      <header>
        <div>
          <strong>{arm.model}</strong>
          <span>{(arm.modalities || ['text']).join(' · ')}</span>
        </div>
        {mixture.fallback_arm_id === arm.id ? (
          <EvaluationTag tone="positive">Fallback</EvaluationTag>
        ) : null}
      </header>
      <dl>
        <div>
          <dt>Quality</dt>
          <dd>
            <MetricReading metric={quality} />
          </dd>
        </div>
        <div>
          <dt>Success</dt>
          <dd>
            <MetricReading metric={success} />
          </dd>
        </div>
        <div>
          <dt>Marginal gain</dt>
          <dd>
            <MetricReading metric={contribution} />
          </dd>
        </div>
      </dl>
      <footer>
        <span>{decisions.join(' · ') || 'Pool-only model'}</span>
        <span>
          ${arm.input_cost_per_million_tokens_usd.toLocaleString()}/M in · $
          {arm.output_cost_per_million_tokens_usd.toLocaleString()}/M out
        </span>
      </footer>
    </article>
  )
}

function MixtureArmMatrix({
  mixture,
  metrics,
}: {
  mixture: EvaluationMixture
  metrics: EvaluationMetric[]
}) {
  const decisionsByArm = decisionNamesByArm(mixture)
  return (
    <div>
      <div className={styles.mixtureSubheading}>
        <div>
          <span>Frozen model pool</span>
          <strong>Per-model outcome matrix</strong>
        </div>
      </div>
      <div className={styles.mixtureArmMatrix}>
        {mixture.model_arms.map((arm) => (
          <MixtureArmCard
            key={arm.id}
            arm={arm}
            metrics={metrics}
            mixture={mixture}
            decisions={decisionsByArm.get(arm.id) || []}
          />
        ))}
      </div>
    </div>
  )
}

function MixtureLineage({ mixture }: { mixture: EvaluationMixture }) {
  return (
    <EvaluationTechnicalDisclosure
      className={styles.mixtureLineageDetails}
      summary="Reproducibility details"
      summaryClassName={styles.mixtureLineageSummary}
    >
      <div className={styles.mixtureLineageStrip}>
        <span>
          Recipe <code>{mixture.recipe_digest}</code>
        </span>
        <span>
          Pool <code>{mixture.pool_digest}</code>
        </span>
        <span>
          Selector <code>{mixture.selector_digest}</code>
        </span>
        <span>
          Adaptation <code>{mixture.adaptation_digest}</code>
        </span>
        <span>
          Binding <code>{mixture.binding_digest}</code>
        </span>
        <span>
          Routing methods{' '}
          <code>
            {mixture.decisions
              .map((decision) => `${decision.name}: ${decision.algorithm}`)
              .join(' · ')}
          </code>
        </span>
      </div>
    </EvaluationTechnicalDisclosure>
  )
}

export default function EvaluationMixtureReport({ report }: { report: EvaluationReport }) {
  const mixture = report.run.mixture
  if (!mixture) return null
  return (
    <section className={layoutStyles.section} aria-labelledby="mixture-report-title">
      <MixtureHeader mixture={mixture} />
      <MixtureOutcomes metrics={report.metrics} mixture={mixture} />
      <div className={styles.mixtureDetailGrid}>
        <MixtureDecisionTopology mixture={mixture} />
        <MixtureArmMatrix mixture={mixture} metrics={report.metrics} />
      </div>
      <MixtureLineage mixture={mixture} />
    </section>
  )
}
