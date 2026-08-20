import { useEffect, useId, useMemo, useRef, useState } from 'react'

import {
  getDefaultMixtureRecipeName,
  getRecipeByName,
  getRecipeNames,
  normalizeEntrypointModelNames,
} from './configPageEntrypointsRecipesSupport'
import type {
  ConfigData,
  DecisionModelRef,
  EntrypointConfig,
  NormalizedModel,
} from './configPageSupport'
import styles from './ConfigPageMixtureDialog.module.css'

interface ConfigPageMixtureDialogProps {
  config: ConfigData
  models: NormalizedModel[]
  entrypoint?: EntrypointConfig
  onClose: () => void
  onSave: (entrypoint: EntrypointConfig) => Promise<void>
}

interface DecisionAssignmentProps {
  name: string
  description?: string
  modelNames: string[]
  selected: Set<string>
  onChange: (selected: Set<string>) => void
  initiallyOpen: boolean
}

function DecisionAssignment({
  name,
  description,
  modelNames,
  selected,
  onChange,
  initiallyOpen,
}: DecisionAssignmentProps) {
  const [search, setSearch] = useState('')
  const query = search.trim().toLowerCase()
  const visible = query
    ? modelNames.filter((model) => model.toLowerCase().includes(query))
    : modelNames
  return (
    <details className={styles.assignment} open={initiallyOpen}>
      <summary>
        <span>
          <strong>{name}</strong>
          <small>{description || 'Decision path'}</small>
        </span>
        <span
          className={`${styles.selectionCount} ${selected.size === 0 ? styles.missingCount : ''}`}
        >
          {selected.size === 0 ? 'Choose models' : `${selected.size} selected`}
        </span>
      </summary>
      <div className={styles.assignmentBody}>
        <div className={styles.pickerToolbar}>
          <input
            type="search"
            value={search}
            onChange={(event) => setSearch(event.target.value)}
            placeholder="Search models"
            aria-label={`Search models for ${name}`}
          />
          <button type="button" onClick={() => onChange(new Set())}>
            Clear
          </button>
        </div>
        <div className={styles.modelPicker}>
          {visible.map((modelName) => (
            <label key={modelName} className={selected.has(modelName) ? styles.selectedModel : ''}>
              <input
                type="checkbox"
                checked={selected.has(modelName)}
                onChange={() => {
                  const next = new Set(selected)
                  if (next.has(modelName)) next.delete(modelName)
                  else next.add(modelName)
                  onChange(next)
                }}
              />
              <span aria-hidden="true">✓</span>
              <code>{modelName}</code>
            </label>
          ))}
          {visible.length === 0 ? <p>No matches</p> : null}
        </div>
      </div>
    </details>
  )
}

export default function ConfigPageMixtureDialog({
  config,
  models,
  entrypoint,
  onClose,
  onSave,
}: ConfigPageMixtureDialogProps) {
  const titleId = useId()
  const dialogRef = useRef<HTMLDivElement>(null)
  const savingRef = useRef(false)
  const recipeNames = getRecipeNames(config)
  const initialRecipeName = entrypoint?.recipe ?? getDefaultMixtureRecipeName(config)
  const [modelName, setModelName] = useState(entrypoint?.model_names[0] ?? '')
  const [aliases, setAliases] = useState(entrypoint?.model_names.slice(1).join('\n') ?? '')
  const [recipeName, setRecipeName] = useState(initialRecipeName)
  const [bindings, setBindings] = useState<Record<string, Set<string>>>(() => {
    if (entrypoint?.model_bindings && Object.keys(entrypoint.model_bindings).length > 0) {
      return Object.fromEntries(
        Object.entries(entrypoint.model_bindings).map(([decision, refs]) => [
          decision,
          new Set(refs.map((ref) => ref.model)),
        ]),
      )
    }
    return Object.fromEntries(
      (getRecipeByName(config, initialRecipeName)?.routing.decisions ?? []).map((decision) => [
        decision.name,
        new Set((decision.modelRefs ?? []).map((reference) => reference.model).filter(Boolean)),
      ]),
    )
  })
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState<string | null>(null)
  savingRef.current = saving
  const recipe = getRecipeByName(config, recipeName)
  const decisions = recipe?.routing.decisions ?? []
  const physicalModelNames = useMemo(() => models.map((model) => model.name).sort(), [models])

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

  const chooseRecipe = (nextRecipe: string) => {
    const next = getRecipeByName(config, nextRecipe)
    setRecipeName(nextRecipe)
    setBindings(
      Object.fromEntries(
        (next?.routing.decisions ?? []).map((decision) => [
          decision.name,
          new Set((decision.modelRefs ?? []).map((reference) => reference.model).filter(Boolean)),
        ]),
      ),
    )
    setError(null)
  }

  const save = async () => {
    const names = normalizeEntrypointModelNames(`${modelName}\n${aliases}`)
    if (names.length === 0) {
      setError('Give this model a name.')
      return
    }
    if (!recipe || decisions.length === 0) {
      setError('Choose a recipe with at least one decision.')
      return
    }
    const missing = decisions.find((decision) => (bindings[decision.name]?.size ?? 0) === 0)
    if (missing) {
      setError(`Choose at least one model for “${missing.name}”.`)
      return
    }
    const modelBindings: Record<string, DecisionModelRef[]> = {}
    for (const decision of decisions) {
      const baseRefs = new Map(
        (decision.modelRefs ?? []).map((reference) => [reference.model, reference]),
      )
      modelBindings[decision.name] = [...bindings[decision.name]].map((name) => ({
        ...(baseRefs.get(name) ?? { model: name, use_reasoning: false }),
        model: name,
      }))
    }
    setSaving(true)
    setError(null)
    try {
      await onSave({ model_names: names, recipe: recipeName, model_bindings: modelBindings })
      onClose()
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Mixture could not be saved.')
    } finally {
      setSaving(false)
    }
  }

  return (
    <div
      className={styles.backdrop}
      onMouseDown={(event) => {
        if (event.target === event.currentTarget && !saving) onClose()
      }}
    >
      <div
        ref={dialogRef}
        className={styles.dialog}
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
        tabIndex={-1}
      >
        <header className={styles.header}>
          <div>
            <span>Mixture-of-Models</span>
            <h2 id={titleId}>{entrypoint ? 'Edit model' : 'Create a model'}</h2>
            <p>Choose a recipe. Assign its models.</p>
          </div>
          <button type="button" onClick={onClose} disabled={saving} aria-label="Close">
            ×
          </button>
        </header>
        {error ? (
          <div className={styles.error} role="alert">
            {error}
          </div>
        ) : null}
        <div className={styles.body}>
          <div className={styles.identityGrid}>
            <label>
              <span>Model name</span>
              <input
                value={modelName}
                onChange={(event) => setModelName(event.target.value)}
                placeholder="my-company/auto"
                autoFocus
              />
            </label>
            <label>
              <span>Recipe</span>
              <select value={recipeName} onChange={(event) => chooseRecipe(event.target.value)}>
                {recipeNames.map((name) => (
                  <option key={name} value={name}>
                    {name}
                  </option>
                ))}
              </select>
            </label>
          </div>

          <section className={styles.decisionSection}>
            <div className={styles.sectionHeader}>
              <div>
                <span>Model assignments</span>
                <strong>{decisions.length} decisions</strong>
              </div>
              <p>{recipe?.description || recipeName}</p>
            </div>
            <div className={styles.assignments}>
              {decisions.map((decision, index) => (
                <DecisionAssignment
                  key={decision.name}
                  name={decision.name}
                  description={decision.description}
                  modelNames={physicalModelNames}
                  selected={bindings[decision.name] ?? new Set()}
                  onChange={(selected) =>
                    setBindings((current) => ({ ...current, [decision.name]: selected }))
                  }
                  initiallyOpen={index === 0 || decisions.length <= 3}
                />
              ))}
              {decisions.length === 0 ? (
                <div className={styles.empty}>This recipe has no decisions.</div>
              ) : null}
            </div>
          </section>

          <details className={styles.advanced}>
            <summary>Aliases</summary>
            <label>
              <span>Additional model names</span>
              <textarea
                value={aliases}
                onChange={(event) => setAliases(event.target.value)}
                placeholder="One per line"
              />
            </label>
          </details>
        </div>
        <footer className={styles.footer}>
          <button type="button" className={styles.cancel} onClick={onClose} disabled={saving}>
            Cancel
          </button>
          <button
            type="button"
            className={styles.save}
            onClick={() => void save()}
            disabled={saving}
          >
            {saving ? 'Saving…' : entrypoint ? 'Save' : 'Create model'}
          </button>
        </footer>
      </div>
    </div>
  )
}
