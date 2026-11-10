import { useId, useMemo, useState } from 'react'

import useAccessibleDialog from '../hooks/useAccessibleDialog'
import ProductIcon from '../components/ProductIcon'
import { type BuiltInRecipeCatalog, composeBuiltInRecipe } from './configPageBuiltInRecipes'
import {
  getPublishableRecipeNames,
  normalizeEntrypointModelNames,
  requireEntrypointModelNames,
  validateEntrypointForm,
} from './configPageEntrypointsRecipesSupport'
import {
  applyRecipeAssignments,
  builtInRecipeBindings,
  builtInRecipeChoices,
  type MixtureRecipeChoice,
  mixtureRecipeKey,
  type ModelAssignments,
  recipeAssignments,
  resolveMixtureRecipe,
} from './configPageMixtureSupport'
import type { ConfigData, EntrypointConfig, NormalizedModel } from './configPageSupport'
import styles from './ConfigPageMixtureDialog.module.css'

const firstDecisionName = (decisionNames: string[]) => decisionNames[0] ?? ''

interface ConfigPageMixtureDialogProps {
  config: ConfigData
  models: NormalizedModel[]
  builtInRecipes?: BuiltInRecipeCatalog | null
  initialRecipe?: MixtureRecipeChoice
  entrypoint?: EntrypointConfig
  entrypointIndex?: number
  onClose: () => void
  onSave: (config: ConfigData) => Promise<void>
}

export default function ConfigPageMixtureDialog({
  config,
  models,
  builtInRecipes = null,
  initialRecipe,
  entrypoint,
  entrypointIndex,
  onClose,
  onSave,
}: ConfigPageMixtureDialogProps) {
  const titleId = useId()
  const recipeNames = getPublishableRecipeNames(config)
  // A built-in recipe publishes a new entrypoint, so editing offers config recipes only.
  const catalog = entrypoint ? null : builtInRecipes
  const builtInChoices = builtInRecipeChoices(catalog)
  const [recipeChoice, setRecipeChoice] = useState<MixtureRecipeChoice | null>(() => {
    const name = entrypoint?.recipe ?? recipeNames[0]
    return initialRecipe ?? (name === undefined ? null : { name })
  })
  const choice = recipeChoice ?? builtInChoices[0] ?? null
  const { description, decisions } = resolveMixtureRecipe(config, catalog, choice)
  const [primaryName, setPrimaryName] = useState(entrypoint?.model_names[0] ?? '')
  const [aliases, setAliases] = useState<string[]>(entrypoint?.model_names.slice(1) ?? [])
  const [aliasDraft, setAliasDraft] = useState('')
  const [assignments, setAssignments] = useState<ModelAssignments>(() =>
    recipeAssignments(config, choice),
  )
  const [activeDecisionName, setActiveDecisionName] = useState(() =>
    firstDecisionName(decisions.map((decision) => decision.name)),
  )
  const [decisionSearch, setDecisionSearch] = useState('')
  const [modelSearch, setModelSearch] = useState('')
  const [showSelectedModels, setShowSelectedModels] = useState(false)
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const dialogRef = useAccessibleDialog<HTMLDivElement>({
    isOpen: true,
    onClose,
    dismissible: !saving,
  })
  const modelOptions = useMemo(
    () =>
      models
        .flatMap((model) => [model.name, ...(model.loras ?? []).map((adapter) => adapter.name)])
        .filter((name, index, names) => names.indexOf(name) === index)
        .sort((left, right) => left.localeCompare(right)),
    [models],
  )

  const chooseRecipe = (key: string) => {
    const nextChoice = [...recipeNames.map((name) => ({ name })), ...builtInChoices].find(
      (candidate) => mixtureRecipeKey(candidate) === key,
    )
    if (!nextChoice) return
    setRecipeChoice(nextChoice)
    setAssignments(recipeAssignments(config, nextChoice))
    setActiveDecisionName(
      firstDecisionName(
        resolveMixtureRecipe(config, catalog, nextChoice).decisions.map(
          (decision) => decision.name,
        ),
      ),
    )
    setDecisionSearch('')
    setModelSearch('')
    setShowSelectedModels(false)
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

  const publishConfigRecipe = (recipeName: string, publicNames: string[]): ConfigData => {
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
    return next
  }

  const save = async () => {
    const publicNames = normalizeEntrypointModelNames([primaryName, ...aliases].join('\n'))
    if (!choice || decisions.length === 0) {
      setError('Choose a recipe with at least one decision.')
      return
    }
    const incompleteDecision = decisions.find(
      (decision) => (assignments[decision.name]?.length ?? 0) < decision.minimumCandidates,
    )
    if (incompleteDecision) {
      const minimum = incompleteDecision.minimumCandidates
      setActiveDecisionName(incompleteDecision.name)
      setError(
        `Choose at least ${minimum} model${minimum === 1 ? '' : 's'} for “${incompleteDecision.name}”.`,
      )
      return
    }

    try {
      setSaving(true)
      setError(null)
      const next =
        choice.bundle === undefined
          ? publishConfigRecipe(choice.name, publicNames)
          : await composeBuiltInRecipe({
              bundle: choice.bundle,
              recipe: choice.name,
              bindings: builtInRecipeBindings(decisions, assignments),
              model_names: requireEntrypointModelNames(publicNames),
            })
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
  const normalizedDecisionSearch = decisionSearch.trim().toLocaleLowerCase()
  const visibleDecisions = normalizedDecisionSearch
    ? decisions.filter((decision) =>
        `${decision.name} ${decision.description ?? ''}`
          .toLocaleLowerCase()
          .includes(normalizedDecisionSearch),
      )
    : decisions
  const normalizedModelSearch = modelSearch.trim().toLocaleLowerCase()
  const visibleModelOptions = modelOptions.filter(
    (modelName) =>
      (!showSelectedModels || activeSelection.includes(modelName)) &&
      (!normalizedModelSearch || modelName.toLocaleLowerCase().includes(normalizedModelSearch)),
  )
  const completedDecisionCount = decisions.filter(
    (decision) => (assignments[decision.name]?.length ?? 0) >= decision.minimumCandidates,
  ).length
  const configRecipeOptions = recipeNames.map((name) => (
    <option key={name} value={mixtureRecipeKey({ name })}>
      {name}
    </option>
  ))

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
              <select
                value={choice ? mixtureRecipeKey(choice) : ''}
                onChange={(event) => chooseRecipe(event.target.value)}
              >
                {catalog?.bundles.length ? (
                  <>
                    {configRecipeOptions.length ? (
                      <optgroup label="Configured recipes">{configRecipeOptions}</optgroup>
                    ) : null}
                    {catalog.bundles.map((bundle) => (
                      <optgroup key={bundle.name} label={`Built-in · ${bundle.name}`}>
                        {bundle.recipes.map((recipe) => (
                          <option
                            key={recipe.name}
                            value={mixtureRecipeKey({ bundle: bundle.name, name: recipe.name })}
                          >
                            {recipe.name}
                          </option>
                        ))}
                      </optgroup>
                    ))}
                  </>
                ) : (
                  configRecipeOptions
                )}
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
                <strong>
                  {completedDecisionCount} of {decisions.length} ready
                </strong>
              </div>
              <p>{description || 'Complete every decision before publishing.'}</p>
            </div>
            {decisions.length > 0 && activeDecision ? (
              <div className={styles.assignmentWorkspace}>
                <div className={styles.decisionRail}>
                  <label className={styles.assignmentSearch}>
                    <ProductIcon name="search" aria-hidden="true" />
                    <input
                      value={decisionSearch}
                      onChange={(event) => setDecisionSearch(event.target.value)}
                      placeholder="Find a decision"
                      aria-label="Find a decision"
                    />
                  </label>
                  <div className={styles.decisionList} role="tablist" aria-label="Recipe decisions">
                    {visibleDecisions.map((decision) => {
                      const decisionIndex = decisions.findIndex(
                        (candidate) => candidate.name === decision.name,
                      )
                      const selected = assignments[decision.name] ?? []
                      const needsModels = decision.minimumCandidates > 0
                      const isActive = decision.name === activeDecision.name
                      return (
                        <button
                          key={decision.name}
                          type="button"
                          role="tab"
                          aria-selected={isActive}
                          className={isActive ? styles.activeDecision : ''}
                          onClick={() => {
                            setActiveDecisionName(decision.name)
                            setModelSearch('')
                            setShowSelectedModels(false)
                          }}
                        >
                          <span className={styles.decisionIndex}>
                            {String(decisionIndex + 1).padStart(2, '0')}
                          </span>
                          <span className={styles.decisionCopy}>
                            <strong>{decision.name}</strong>
                            <small>{decision.description || 'Model path'}</small>
                          </span>
                          <span
                            className={
                              selected.length || !needsModels
                                ? styles.selectionCount
                                : styles.missingCount
                            }
                          >
                            {needsModels ? selected.length || '—' : 'No model'}
                          </span>
                        </button>
                      )
                    })}
                    {visibleDecisions.length === 0 ? (
                      <p className={styles.noResults}>No matching decisions.</p>
                    ) : null}
                  </div>
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
                      {activeDecision.minimumCandidates > 0
                        ? `${activeSelection.length} selected · minimum ${activeDecision.minimumCandidates}`
                        : 'No model needed'}
                    </span>
                  </div>
                  {activeDecision.minimumCandidates > 0 ? (
                    <>
                      <div className={styles.modelPickerToolbar}>
                        <label className={styles.assignmentSearch}>
                          <ProductIcon name="search" aria-hidden="true" />
                          <input
                            value={modelSearch}
                            onChange={(event) => setModelSearch(event.target.value)}
                            placeholder="Find a model"
                            aria-label={`Find a model for ${activeDecision.name}`}
                          />
                        </label>
                        <button
                          type="button"
                          className={showSelectedModels ? styles.activeFilter : ''}
                          aria-pressed={showSelectedModels}
                          onClick={() => setShowSelectedModels((current) => !current)}
                        >
                          <ProductIcon name="check" aria-hidden="true" />
                          Selected
                          <span>{activeSelection.length}</span>
                        </button>
                      </div>
                      <div className={styles.modelPicker}>
                        {visibleModelOptions.map((modelName) => (
                          <label
                            key={modelName}
                            className={
                              activeSelection.includes(modelName) ? styles.selectedModel : ''
                            }
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
                        {visibleModelOptions.length === 0 ? (
                          <p>
                            {modelOptions.length === 0
                              ? 'Connect a model before publishing this mixture.'
                              : 'No models match this view.'}
                          </p>
                        ) : null}
                      </div>
                    </>
                  ) : (
                    <p className={styles.empty}>This decision responds without calling a model.</p>
                  )}
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
