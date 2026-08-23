import useAccessibleDialog from '../hooks/useAccessibleDialog'
import ProductIcon from '../components/ProductIcon'
import type {
  RoutingEntrypoint,
  RoutingModelCardView,
  RoutingRecipe,
} from '../utils/routingManagementApi'
import styles from './ConfigPageMoMTopologyDialog.module.css'

interface Props {
  entrypoint: RoutingEntrypoint
  recipes: RoutingRecipe[]
  models: RoutingModelCardView[]
  canManage: boolean
  pending: boolean
  error?: string | null
  onClose: () => void
  onEdit: () => void
  onPublish: () => void
  onDelete: () => void
}

export default function ConfigPageMoMTopologyDialog({
  entrypoint,
  recipes,
  models,
  canManage,
  pending,
  error,
  onClose,
  onEdit,
  onPublish,
  onDelete,
}: Props) {
  const dialogRef = useAccessibleDialog<HTMLDivElement>({ isOpen: true, onClose })
  const displayName = entrypoint.aliases[0] ?? entrypoint.name
  const modelById = new Map(models.map((model) => [model.id, model]))
  const recipeById = new Map(recipes.map((recipe) => [recipe.id, recipe]))
  const ruleViews = (entrypoint.rules ?? []).map((rule) => ({
    rule,
    recipe: recipeById.get(rule.recipeId),
  }))
  const decisionViews = ruleViews.flatMap(({ rule, recipe }) => {
    const decisionById = new Map(
      (recipe?.decisions ?? []).map((decision) => [decision.id, decision]),
    )
    const decisionIds = [
      ...new Set([
        ...(recipe?.decisions.map((decision) => decision.id) ?? []),
        ...Object.keys(rule.assignments),
      ]),
    ]
    return decisionIds.map((decisionId) => ({
      rule,
      decisionId,
      decision: decisionById.get(decisionId),
      assignment: rule.assignments[decisionId],
    }))
  })
  const assignedIds = [
    ...new Set(
      ruleViews.flatMap(({ rule }) =>
        Object.values(rule.assignments).flatMap((assignment) =>
          assignment.models.map((model) => model.modelId),
        ),
      ),
    ),
  ]

  return (
    <div className={styles.overlay} role="presentation" onMouseDown={onClose}>
      <div
        ref={dialogRef}
        className={styles.dialog}
        role="dialog"
        aria-modal="true"
        aria-label={`Topology for ${displayName}`}
        tabIndex={-1}
        onMouseDown={(event) => event.stopPropagation()}
      >
        <header className={styles.header}>
          <div className={styles.heading}>
            <ProductIcon name="topology" aria-hidden="true" />
            <div>
              <span>Mixture topology</span>
              <h2>{displayName}</h2>
              <p>Models assigned to each decision.</p>
            </div>
          </div>
          <button type="button" onClick={onClose} aria-label="Close topology">
            <ProductIcon name="close" />
          </button>
        </header>
        {error ? (
          <div className={styles.error} role="alert">
            <ProductIcon name="alert" aria-hidden="true" />
            <span>{error}</span>
          </div>
        ) : null}
        <div className={styles.canvas}>
          <section className={styles.stage}>
            <span className={styles.stageLabel}>Model</span>
            <div className={`${styles.node} ${styles.entrypointNode}`}>
              {(entrypoint.aliases.length > 0 ? entrypoint.aliases : [entrypoint.name]).map(
                (alias) => (
                  <code key={alias}>{alias}</code>
                ),
              )}
            </div>
          </section>
          <span className={styles.connector} aria-hidden="true">
            <ProductIcon name="chevron-right" />
          </span>
          <section className={styles.stage}>
            <span className={styles.stageLabel}>Rules</span>
            <div className={styles.nodeList}>
              {ruleViews.map(({ rule, recipe }) => (
                <div key={rule.id} className={`${styles.node} ${styles.recipeNode}`}>
                  <strong>{recipe?.name ?? rule.recipeId}</strong>
                  <small>
                    {rule.name} · {recipe?.decisions.length ?? Object.keys(rule.assignments).length}{' '}
                    decisions
                  </small>
                </div>
              ))}
              {ruleViews.length === 0 ? (
                <div className={`${styles.node} ${styles.emptyNode}`}>No rules</div>
              ) : null}
            </div>
          </section>
          <span className={styles.connector} aria-hidden="true">
            <ProductIcon name="chevron-right" />
          </span>
          <section className={`${styles.stage} ${styles.wideStage}`}>
            <span className={styles.stageLabel}>Decisions</span>
            <div className={styles.nodeList}>
              {decisionViews.map(({ rule, decisionId, decision, assignment }) => (
                <div
                  key={`${rule.id}-${decisionId}`}
                  className={`${styles.node} ${styles.decisionNode}`}
                >
                  <strong>{decision?.name ?? decisionId}</strong>
                  <small>
                    {rule.name} ·{' '}
                    {decision?.dispatchCardinality === 'multi' ? 'Multi-model' : 'Single model'}
                  </small>
                  <div className={styles.decisionTargets}>
                    {(assignment?.models ?? []).map((reference) => (
                      <code
                        key={`${rule.id}-${decisionId}-${reference.modelId}-${reference.priority}`}
                      >
                        {modelById.get(reference.modelId)?.name ?? reference.modelId}
                        {reference.priority > 0 ? ` · P${reference.priority}` : ''}
                      </code>
                    ))}
                    {!assignment?.models.length ? <small>Not assigned</small> : null}
                  </div>
                </div>
              ))}
              {decisionViews.length === 0 ? (
                <div className={`${styles.node} ${styles.emptyNode}`}>No decisions</div>
              ) : null}
            </div>
          </section>
          <span className={styles.connector} aria-hidden="true">
            <ProductIcon name="chevron-right" />
          </span>
          <section className={`${styles.stage} ${styles.wideStage}`}>
            <span className={styles.stageLabel}>Models</span>
            <div className={styles.nodeList}>
              {assignedIds.map((id) => (
                <div key={id} className={`${styles.node} ${styles.modelNode}`}>
                  <ProductIcon name="model" aria-hidden="true" />
                  <code>{modelById.get(id)?.name ?? id}</code>
                </div>
              ))}
              {assignedIds.length === 0 ? (
                <div className={`${styles.node} ${styles.emptyNode}`}>No models assigned</div>
              ) : null}
            </div>
          </section>
        </div>
        {canManage ? (
          <footer className={styles.actions}>
            <button type="button" onClick={onDelete} disabled={pending}>
              <ProductIcon name="trash" />
              Delete
            </button>
            <div>
              <button type="button" onClick={onEdit} disabled={pending}>
                <ProductIcon name="edit" />
                Edit
              </button>
              <button type="button" onClick={onPublish} disabled={pending}>
                <ProductIcon name={entrypoint.status === 'active' ? 'power' : 'check'} />
                {pending ? 'Working…' : entrypoint.status === 'active' ? 'Unpublish' : 'Publish'}
              </button>
            </div>
          </footer>
        ) : null}
      </div>
    </div>
  )
}
