import { useId, useState } from 'react'

import ProductIcon from '../components/ProductIcon'
import useAccessibleDialog from '../hooks/useAccessibleDialog'
import type { RoutingRecipe, RoutingRecipeWrite } from '../utils/routingManagementApi'
import shellStyles from './ConfigPageMixtureDialog.module.css'
import styles from './ConfigPageRecipeDialog.module.css'
import {
  recipeDocumentSummary,
  recipeWrite,
  EMPTY_RECIPE_DOCUMENT,
  suggestedRecipeCopyName,
} from './configPageRecipeDialogSupport'

interface Props {
  recipe?: RoutingRecipe
  duplicateFrom?: RoutingRecipe
  readOnly?: boolean
  onClose: () => void
  onSave: (input: RoutingRecipeWrite) => Promise<void>
  onDelete?: () => void
  onDuplicate?: () => void
}

export default function ConfigPageRecipeDialog({
  recipe,
  duplicateFrom,
  readOnly = false,
  onClose,
  onSave,
  onDelete,
  onDuplicate,
}: Props) {
  const titleId = useId()
  const sourceRecipe = duplicateFrom ?? recipe
  const sourceDocument = sourceRecipe?.document ?? EMPTY_RECIPE_DOCUMENT
  const summary = sourceRecipe ? recipeDocumentSummary(sourceRecipe) : null
  const [name, setName] = useState(
    duplicateFrom ? suggestedRecipeCopyName(duplicateFrom) : (recipe?.name ?? ''),
  )
  const [description, setDescription] = useState(sourceRecipe?.description ?? '')
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const dialogRef = useAccessibleDialog<HTMLDivElement>({
    isOpen: true,
    onClose,
    dismissible: !saving,
  })

  const save = async () => {
    const trimmedName = name.trim()
    if (!trimmedName || trimmedName.length > 256) {
      setError('Name is required and must be 256 characters or fewer.')
      return
    }
    setSaving(true)
    setError(null)
    try {
      await onSave(recipeWrite(trimmedName, description, sourceDocument))
      onClose()
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Recipe could not be saved.')
    } finally {
      setSaving(false)
    }
  }

  return (
    <div
      className={shellStyles.backdrop}
      onMouseDown={(event) => {
        if (event.target === event.currentTarget && !saving) onClose()
      }}
    >
      <div
        ref={dialogRef}
        className={`${shellStyles.dialog} ${styles.dialog}`}
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
        aria-busy={saving}
        tabIndex={-1}
      >
        <header className={shellStyles.header}>
          <div>
            <span>Recipe</span>
            <h2 id={titleId}>
              {duplicateFrom ? 'Duplicate recipe' : recipe ? recipe.name : 'Create a recipe'}
            </h2>
            <p>
              {readOnly
                ? 'Signals, projections, and decisions.'
                : duplicateFrom
                  ? 'Start from a proven design. Make it yours.'
                  : 'Name the design. Build it in Signals, Projections, and Decisions.'}
            </p>
          </div>
          <button type="button" onClick={onClose} disabled={saving} aria-label="Close">
            <ProductIcon name="close" />
          </button>
        </header>
        {error ? (
          <div className={shellStyles.error} role="alert">
            {error}
          </div>
        ) : null}
        <div className={`${shellStyles.body} ${styles.body}`}>
          <div className={styles.designPane}>
            <div className={styles.identityGrid}>
              <label>
                <span>Name</span>
                <input
                  value={name}
                  onChange={(event) => setName(event.target.value)}
                  disabled={readOnly || (Boolean(recipe) && !duplicateFrom)}
                  placeholder="Quality first"
                  data-dialog-initial-focus={!recipe || Boolean(duplicateFrom) ? '' : undefined}
                />
              </label>
              <label>
                <span>Description</span>
                <input
                  value={description}
                  onChange={(event) => setDescription(event.target.value)}
                  disabled={readOnly}
                  placeholder="When quality matters most"
                />
              </label>
            </div>
            {summary ? (
              <>
                <div className={styles.recipeSummary} aria-label="Recipe contents">
                  <div>
                    <strong>{summary.signals}</strong>
                    <span>Signals</span>
                  </div>
                  <div>
                    <strong>{summary.projections}</strong>
                    <span>Projections</span>
                  </div>
                  <div>
                    <strong>{summary.decisions}</strong>
                    <span>Decisions</span>
                  </div>
                </div>
                <div className={styles.decisionList}>
                  {sourceRecipe?.decisions.map((decision) => (
                    <div key={decision.id}>
                      <strong>{decision.name}</strong>
                      <span>
                        {decision.dispatchCardinality === 'single'
                          ? 'Single model path'
                          : 'Multi-model path'}
                      </span>
                    </div>
                  ))}
                </div>
              </>
            ) : (
              <div className={styles.nextStep}>
                <ProductIcon name="mixture" />
                <div>
                  <strong>Build after creation</strong>
                  <span>Add Signals, Projections, and Decisions from their dedicated pages.</span>
                </div>
              </div>
            )}
          </div>
        </div>
        <footer className={shellStyles.footer}>
          {onDelete ? (
            <button
              type="button"
              className={shellStyles.cancel}
              onClick={onDelete}
              disabled={saving}
            >
              <ProductIcon name="trash" />
              Delete
            </button>
          ) : null}
          {onDuplicate ? (
            <button
              type="button"
              className={shellStyles.cancel}
              onClick={onDuplicate}
              disabled={saving}
            >
              <ProductIcon name="copy" />
              Duplicate
            </button>
          ) : null}
          <button type="button" className={shellStyles.cancel} onClick={onClose} disabled={saving}>
            <ProductIcon name={readOnly ? 'check' : 'close'} />
            {readOnly ? 'Done' : 'Cancel'}
          </button>
          {!readOnly ? (
            <button
              type="button"
              className={shellStyles.save}
              onClick={() => void save()}
              disabled={saving}
            >
              <ProductIcon name={recipe ? 'check' : 'plus'} />
              {saving
                ? 'Saving…'
                : duplicateFrom
                  ? 'Duplicate recipe'
                  : recipe
                    ? 'Save'
                    : 'Create recipe'}
            </button>
          ) : null}
        </footer>
      </div>
    </div>
  )
}
