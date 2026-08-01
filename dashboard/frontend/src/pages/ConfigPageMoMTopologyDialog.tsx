import useAccessibleDialog from '../hooks/useAccessibleDialog'
import { collectRecipeTargetModels } from './configPageEntrypointsRecipesSupport'
import type { EntrypointConfig, RecipeConfig } from './configPageSupport'
import styles from './ConfigPageMoMTopologyDialog.module.css'

interface ConfigPageMoMTopologyDialogProps {
  entrypoint: EntrypointConfig
  recipe: RecipeConfig
  onClose: () => void
}

export default function ConfigPageMoMTopologyDialog({
  entrypoint,
  recipe,
  onClose,
}: ConfigPageMoMTopologyDialogProps) {
  const dialogRef = useAccessibleDialog<HTMLDivElement>({
    isOpen: true,
    onClose,
  })
  const decisions = recipe.routing.decisions ?? []
  const models = collectRecipeTargetModels(recipe)

  return (
    <div className={styles.overlay} role="presentation" onMouseDown={onClose}>
      <div
        ref={dialogRef}
        className={styles.dialog}
        role="dialog"
        aria-modal="true"
        aria-label={`Topology for ${entrypoint.model_names.join(', ')}`}
        tabIndex={-1}
        onMouseDown={(event) => event.stopPropagation()}
      >
        <header className={styles.header}>
          <div>
            <span>Mixture topology</span>
            <h2>{entrypoint.model_names.join(', ')}</h2>
            <p>How the public model resolves into routing decisions and AMD model pools.</p>
          </div>
          <button type="button" onClick={onClose} aria-label="Close topology">
            ×
          </button>
        </header>

        <div className={styles.canvas}>
          <section className={styles.stage}>
            <span className={styles.stageLabel}>Entrypoint</span>
            <div className={`${styles.node} ${styles.entrypointNode}`}>
              {entrypoint.model_names.map((model) => (
                <code key={model}>{model}</code>
              ))}
            </div>
          </section>

          <span className={styles.connector} aria-hidden="true">
            →
          </span>

          <section className={styles.stage}>
            <span className={styles.stageLabel}>Recipe</span>
            <div className={`${styles.node} ${styles.recipeNode}`}>
              <strong>{recipe.name}</strong>
              <small>{decisions.length} decision lanes</small>
            </div>
          </section>

          <span className={styles.connector} aria-hidden="true">
            →
          </span>

          <section className={`${styles.stage} ${styles.wideStage}`}>
            <span className={styles.stageLabel}>Decisions</span>
            <div className={styles.nodeList}>
              {decisions.map((decision) => (
                <div key={decision.name} className={`${styles.node} ${styles.decisionNode}`}>
                  <strong>{decision.name}</strong>
                  <small>
                    P{decision.priority} · {decision.modelRefs?.length ?? 0} models
                  </small>
                </div>
              ))}
            </div>
          </section>

          <span className={styles.connector} aria-hidden="true">
            →
          </span>

          <section className={`${styles.stage} ${styles.wideStage}`}>
            <span className={styles.stageLabel}>Model pool</span>
            <div className={styles.nodeList}>
              {models.map((model) => (
                <div key={model} className={`${styles.node} ${styles.modelNode}`}>
                  <span className={styles.modelDot} aria-hidden="true" />
                  <code>{model}</code>
                </div>
              ))}
            </div>
          </section>
        </div>
      </div>
    </div>
  )
}
