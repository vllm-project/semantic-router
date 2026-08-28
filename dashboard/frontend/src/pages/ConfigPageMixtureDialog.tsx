import { useId, useMemo, useState } from 'react'

import useAccessibleDialog from '../hooks/useAccessibleDialog'
import ProductIcon from '../components/ProductIcon'
import {
  DEFAULT_RECIPE_NAME,
  getPublishableRecipeNames,
  getRecipeByName,
  getRecipeNames,
  normalizeEntrypointModelNames,
  validateEntrypointForm,
} from './configPageEntrypointsRecipesSupport'
import {
  applyRecipeAssignments,
  assignmentState,
  type ModelAssignments,
} from './configPageMixtureSupport'
import type { ConfigData, EntrypointConfig, NormalizedModel } from './configPageSupport'
import styles from './ConfigPageMixtureDialog.module.css'

const firstDecisionName = (decisionNames: string[]) => decisionNames[0] ?? ''

interface ConfigPageMixtureDialogProps {
  config: ConfigData
  models: NormalizedModel[]
  entrypoint?: EntrypointConfig
  entrypointIndex?: number
  onClose: () => void
  onSave: (config: ConfigData) => Promise<void>
}

export default function ConfigPageMixtureDialog({
  config,
  models,
  entrypoint,
  entrypointIndex,
  onClose,
  onSave,
}: ConfigPageMixtureDialogProps) {
  const titleId = useId()
  const recipeNames = getPublishableRecipeNames(config)
  const initialRecipeName =
    entrypoint?.recipe ?? recipeNames[0] ?? getRecipeNames(config)[0] ?? DEFAULT_RECIPE_NAME
  const initialRecipe = getRecipeByName(config, initialRecipeName)
  const [primaryName, setPrimaryName] = useState(entrypoint?.model_names[0] ?? '')
  const [aliases, setAliases] = useState<string[]>(entrypoint?.model_names.slice(1) ?? [])
  const [aliasDraft, setAliasDraft] = useState('')
  const [recipeName, setRecipeName] = useState(initialRecipeName)
  const [assignments, setAssignments] = useState<ModelAssignments>(() =>
    assignmentState(initialRecipe?.routing.decisions ?? []),
  )
  const [activeDecisionName, setActiveDecisionName] = useState(() =>
    firstDecisionName((initialRecipe?.routing.decisions ?? []).map((decision) => decision.name)),
  )
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const dialogRef = useAccessibleDialog<HTMLDivElement>({
    isOpen: true,
    onClose,
    dismissible: !saving,
  })
  const recipe = getRecipeByName(config, recipeName)
  const decisions = recipe?.routing.decisions ?? []
  const modelOptions = useMemo(
    () =>
      models
        .flatMap((model) => [model.name, ...(model.loras ?? []).map((adapter) => adapter.name)])
        .filter((name, index, names) => names.indexOf(name) === index)
        .sort((left, right) => left.localeCompare(right)),
    [models],
  )

  const chooseRecipe = (nextRecipeName: string) => {
    const nextRecipe = getRecipeByName(config, nextRecipeName)
    setRecipeName(nextRecipeName)
    setAssignments(assignmentState(nextRecipe?.routing.decisions ?? []))
    setActiveDecisionName(
      firstDecisionName((nextRecipe?.routing.decisions ?? []).map((decision) => decision.name)),
    )
    setError(null)
  }

  const toggleModel = (decisionName: string, modelName: string) => {
    setAssignments((current) => {
      const selected = current[decisionName] ?? []
      return {
        ...current,
        [decisionName]: selected.includes(modelName)
          ? selected.filter((name) => name !== modelName)
          : [...selected, modelName],
      }
    })
  }

  const addAlias = () => {
    const alias = aliasDraft.trim()
    if (!alias || alias === primaryName.trim() || aliases.includes(alias)) return
    setAliases((current) => [...current, alias])
    setAliasDraft('')
  }

  const save = async () => {
    const publicNames = normalizeEntrypointModelNames([primaryName, ...aliases].join('\n'))
    if (!recipe || decisions.length === 0) {
      setError('Choose a recipe with at least one decision.')
      return
    }
    const incompleteDecision = decisions.find(
      (decision) => (assignments[decision.name]?.length ?? 0) === 0,
    )
    if (incompleteDecision) {
      setActiveDecisionName(incompleteDecision.name)
      setError(`Choose at least one model for “${incompleteDecision.name}”.`)
      return
    }

    try {
      const normalizedEntrypoint = validateEntrypointForm(
        { modelNames: publicNames.join('\n'), recipe: recipeName },
        config,
        models,
        entrypointIndex ?? null,
      )
      const next = applyRecipeAssignments(config, recipeName, assignments)
      const entrypoints = [...(next.entrypoints ?? [])]
      if (entrypointIndex === undefined) entrypoints.push(normalizedEntrypoint)
      else entrypoints[entrypointIndex] = normalizedEntrypoint
      next.entrypoints = entrypoints

      setSaving(true)
      setError(null)
      await onSave(next)
      onClose()
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Model could not be saved.')
    } finally {
      setSaving(false)
    }
  }

  const activeDecision =
    decisions.find((decision) => decision.name === activeDecisionName) ?? decisions[0]
  const activeSelection = activeDecision ? (assignments[activeDecision.name] ?? []) : []

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
        aria-busy={saving}
        tabIndex={-1}
      >
        <header className={styles.header}>
          <div className={styles.headerIdentity}>
            <div className={styles.headerLogo} aria-hidden="true">
              <img src="/vllm.png" alt="" />
            </div>
            <div>
              <span>Mixture-of-Models</span>
              <h2 id={titleId}>{entrypoint ? 'Edit model' : 'Create model'}</h2>
              <p>Choose a recipe. Assign the right models.</p>
            </div>
          </div>
          <button type="button" onClick={onClose} disabled={saving} aria-label="Close dialog">
            <ProductIcon name="close" />
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
                value={primaryName}
                onChange={(event) => setPrimaryName(event.target.value)}
                placeholder="my-company/auto"
                data-dialog-initial-focus
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

          <section className={styles.aliasSection}>
            <div className={styles.aliasHeading}>
              <div>
                <span>Aliases</span>
                <strong>Additional names</strong>
              </div>
              <p>Optional names that call this same model.</p>
            </div>
            <div className={styles.aliasWorkspace}>
              <div className={styles.aliasComposer}>
                <input
                  value={aliasDraft}
                  onChange={(event) => setAliasDraft(event.target.value)}
                  onKeyDown={(event) => {
                    if (event.key === 'Enter') {
                      event.preventDefault()
                      addAlias()
                    }
                  }}
                  placeholder="Add an alias"
                />
                <button type="button" onClick={addAlias} disabled={!aliasDraft.trim()}>
                  <ProductIcon name="plus" aria-hidden="true" />
                  Add
                </button>
              </div>
              {aliases.length > 0 ? (
                <div className={styles.aliasList}>
                  {aliases.map((alias) => (
                    <div key={alias} className={styles.aliasItem}>
                      <code>{alias}</code>
                      <button
                        type="button"
                        aria-label={`Remove alias ${alias}`}
                        onClick={() =>
                          setAliases((current) => current.filter((item) => item !== alias))
                        }
                      >
                        <ProductIcon name="close" />
                      </button>
                    </div>
                  ))}
                </div>
              ) : null}
            </div>
          </section>

          <section className={styles.decisionSection}>
            <div className={styles.sectionHeader}>
              <div>
                <span>Model assignments</span>
                <strong>{decisions.length} decisions</strong>
              </div>
              <p>{recipe?.description || 'Complete every decision before publishing.'}</p>
            </div>
            {decisions.length > 0 && activeDecision ? (
              <div className={styles.assignmentWorkspace}>
                <div className={styles.decisionRail} role="tablist" aria-label="Recipe decisions">
                  {decisions.map((decision, decisionIndex) => {
                    const selected = assignments[decision.name] ?? []
                    const isActive = decision.name === activeDecision.name
                    return (
                      <button
                        key={decision.name}
                        type="button"
                        role="tab"
                        aria-selected={isActive}
                        className={isActive ? styles.activeDecision : ''}
                        onClick={() => setActiveDecisionName(decision.name)}
                      >
                        <span className={styles.decisionIndex}>
                          {String(decisionIndex + 1).padStart(2, '0')}
                        </span>
                        <span className={styles.decisionCopy}>
                          <strong>{decision.name}</strong>
                          <small>{decision.description || 'Model path'}</small>
                        </span>
                        <span
                          className={selected.length ? styles.selectionCount : styles.missingCount}
                        >
                          {selected.length || '—'}
                        </span>
                      </button>
                    )
                  })}
                </div>

                <div
                  className={styles.assignmentPanel}
                  role="tabpanel"
                  aria-label={`${activeDecision.name} model assignment`}
                >
                  <div className={styles.assignmentPanelHeader}>
                    <div>
                      <strong>{activeDecision.name}</strong>
                      <span>
                        {activeDecision.description || 'Choose the models for this path.'}
                      </span>
                    </div>
                    <span>
                      {activeSelection.length
                        ? `${activeSelection.length} selected`
                        : 'Choose at least one'}
                    </span>
                  </div>
                  <div className={styles.modelPicker}>
                    {modelOptions.map((modelName) => (
                      <label
                        key={modelName}
                        className={activeSelection.includes(modelName) ? styles.selectedModel : ''}
                      >
                        <input
                          type="checkbox"
                          checked={activeSelection.includes(modelName)}
                          onChange={() => toggleModel(activeDecision.name, modelName)}
                        />
                        <span aria-hidden="true">
                          <ProductIcon name="check" />
                        </span>
                        <code>{modelName}</code>
                      </label>
                    ))}
                    {modelOptions.length === 0 ? (
                      <p>Connect a model before publishing this mixture.</p>
                    ) : null}
                  </div>
                </div>
              </div>
            ) : (
              <div className={styles.empty}>Add decisions to this recipe first.</div>
            )}
          </section>
        </div>

        <footer className={styles.footer}>
          <button type="button" className={styles.cancel} onClick={onClose} disabled={saving}>
            <ProductIcon name="close" aria-hidden="true" />
            Cancel
          </button>
          <button
            type="button"
            className={styles.save}
            onClick={() => void save()}
            disabled={saving}
          >
            <ProductIcon name={entrypoint ? 'check' : 'plus'} aria-hidden="true" />
            {saving ? 'Saving…' : entrypoint ? 'Save model' : 'Create model'}
          </button>
        </footer>
      </div>
    </div>
  )
}
