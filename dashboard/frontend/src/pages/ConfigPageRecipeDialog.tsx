import { useEffect, useId, useRef, useState } from 'react'

import type { RecipeConfig } from './configPageSupport'
import shellStyles from './ConfigPageMixtureDialog.module.css'
import styles from './ConfigPageRecipeDialog.module.css'

interface Props {
  recipe?: RecipeConfig
  published?: boolean
  onClose: () => void
  onSave: (
    identity: { name: string; description: string },
    originalName: string | null,
  ) => Promise<void>
}

export default function ConfigPageRecipeDialog({
  recipe,
  published = false,
  onClose,
  onSave,
}: Props) {
  const titleId = useId()
  const dialogRef = useRef<HTMLDivElement>(null)
  const savingRef = useRef(false)
  const [name, setName] = useState(recipe?.name ?? '')
  const [description, setDescription] = useState(recipe?.description ?? '')
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState<string | null>(null)
  savingRef.current = saving

  useEffect(() => {
    const previous = document.activeElement as HTMLElement | null
    dialogRef.current?.focus()
    const onKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape' && !savingRef.current) onClose()
    }
    document.addEventListener('keydown', onKey)
    return () => {
      document.removeEventListener('keydown', onKey)
      previous?.focus()
    }
  }, [onClose])

  const save = async () => {
    setSaving(true)
    setError(null)
    try {
      await onSave({ name, description }, recipe?.name ?? null)
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
        tabIndex={-1}
      >
        <header className={shellStyles.header}>
          <div>
            <span>Recipe</span>
            <h2 id={titleId}>{recipe ? 'Edit recipe' : 'Create a recipe'}</h2>
            <p>Name the path. Build it from Signals, Projections, and Decisions.</p>
          </div>
          <button type="button" onClick={onClose} disabled={saving} aria-label="Close">
            ×
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
                  placeholder="quality-first"
                  disabled={published}
                  autoFocus={!published}
                />
              </label>
              <label>
                <span>Description</span>
                <input
                  value={description}
                  onChange={(event) => setDescription(event.target.value)}
                  placeholder="When quality matters most"
                  autoFocus={published}
                />
              </label>
            </div>
            <div className={styles.recipeSummary}>
              <div>
                <strong>1</strong>
                <span>Create Recipe</span>
              </div>
              <div>
                <strong>2</strong>
                <span>Build its path</span>
              </div>
              <div>
                <strong>3</strong>
                <span>Publish a model</span>
              </div>
            </div>
          </div>
        </div>
        <footer className={shellStyles.footer}>
          <button type="button" className={shellStyles.cancel} onClick={onClose} disabled={saving}>
            Cancel
          </button>
          <button
            type="button"
            className={shellStyles.save}
            onClick={() => void save()}
            disabled={saving}
          >
            {saving ? 'Saving…' : recipe ? 'Save' : 'Create recipe'}
          </button>
        </footer>
      </div>
    </div>
  )
}
