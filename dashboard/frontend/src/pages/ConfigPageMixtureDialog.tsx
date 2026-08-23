import { useId, useMemo, useState } from 'react'

import ProductIcon from '../components/ProductIcon'
import useAccessibleDialog from '../hooks/useAccessibleDialog'
import type {
  RoutingAssignmentModelWrite,
  RoutingAssignmentSetWrite,
  RoutingEntrypoint,
  RoutingEntrypointRule,
  RoutingEntrypointWrite,
  RoutingModelCardView,
  RoutingRecipe,
} from '../utils/routingManagementApi'
import styles from './ConfigPageMixtureDialog.module.css'

interface Props {
  models: RoutingModelCardView[]
  recipes: RoutingRecipe[]
  entrypoint?: RoutingEntrypoint
  onClose: () => void
  onSave: (entrypoint: RoutingEntrypointWrite) => Promise<void>
}

const editableAssignments = (rule?: RoutingEntrypointRule) =>
  Object.fromEntries(
    Object.entries(rule?.assignments ?? {}).map(([decisionId, assignmentSet]) => [
      decisionId,
      assignmentSet.models.map(({ modelId, priority, weight, loraName, reasoning }) => ({
        modelId,
        priority,
        weight,
        ...(loraName ? { loraName } : {}),
        ...(reasoning ? { reasoning } : {}),
      })),
    ]),
  ) as Record<string, RoutingAssignmentModelWrite[]>

const editableFallbacks = (rule?: RoutingEntrypointRule) =>
  Object.fromEntries(
    Object.entries(rule?.assignments ?? {})
      .filter(([, assignmentSet]) => assignmentSet.fallback)
      .map(([decisionId, assignmentSet]) => [decisionId, assignmentSet.fallback]),
  ) as Record<string, RoutingAssignmentSetWrite['fallback']>

function assignmentRuleWrite(rule: RoutingEntrypointRule) {
  return {
    id: rule.id,
    name: rule.name,
    matchers: rule.matchers,
    recipeId: rule.recipeId,
    assignments: Object.fromEntries(
      Object.entries(rule.assignments).map(([decisionId, assignmentSet]) => [
        decisionId,
        {
          models: assignmentSet.models.map(
            ({ modelId, priority, weight, loraName, reasoning }) => ({
              modelId,
              priority,
              weight,
              loraName,
              reasoning,
            }),
          ),
          fallback: assignmentSet.fallback,
        },
      ]),
    ),
  }
}

export default function ConfigPageMixtureDialog({
  models,
  recipes,
  entrypoint,
  onClose,
  onSave,
}: Props) {
  const titleId = useId()
  const initialRule = entrypoint?.rules?.[0]
  const [modelName, setModelName] = useState(entrypoint?.aliases[0] ?? entrypoint?.name ?? '')
  const [aliases, setAliases] = useState(entrypoint?.aliases.slice(1).join('\n') ?? '')
  const [recipeId, setRecipeId] = useState(initialRule?.recipeId ?? recipes[0]?.id ?? '')
  const [assignments, setAssignments] = useState<Record<string, RoutingAssignmentModelWrite[]>>(
    () => editableAssignments(initialRule),
  )
  const [fallbacks, setFallbacks] = useState<Record<string, RoutingAssignmentSetWrite['fallback']>>(
    () => editableFallbacks(initialRule),
  )
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const dialogRef = useAccessibleDialog<HTMLDivElement>({
    isOpen: true,
    onClose,
    dismissible: !saving,
  })
  const recipe = recipes.find((candidate) => candidate.id === recipeId)
  const orderedModels = useMemo(
    () => [...models].sort((a, b) => a.name.localeCompare(b.name)),
    [models],
  )

  const chooseRecipe = (nextRecipeId: string) => {
    setRecipeId(nextRecipeId)
    const next = recipes.find((candidate) => candidate.id === nextRecipeId)
    setAssignments(Object.fromEntries((next?.decisions ?? []).map((decision) => [decision.id, []])))
    setFallbacks({})
    setError(null)
  }

  const toggleModel = (decisionId: string, modelId: string) => {
    const values = assignments[decisionId] ?? []
    const selected = values.some((value) => value.modelId === modelId)
    const fallbackEnabled = Boolean(fallbacks[decisionId])
    let next = selected
      ? values.filter((value) => value.modelId !== modelId)
      : [...values, { modelId, priority: fallbackEnabled ? 1 : 0, weight: '1' }]
    if (fallbackEnabled && next.length < 2) {
      next = next.map((value) => ({ ...value, priority: 0 }))
      setFallbacks((current) => ({ ...current, [decisionId]: undefined }))
    }
    setAssignments((current) => ({ ...current, [decisionId]: next }))
  }

  const setFallback = (decisionId: string, enabled: boolean) => {
    setAssignments((current) => ({
      ...current,
      [decisionId]: (current[decisionId] ?? []).map((value, index) => ({
        ...value,
        priority: enabled && index > 0 ? 1 : 0,
      })),
    }))
    setFallbacks((current) => ({
      ...current,
      [decisionId]: enabled
        ? { strategy: 'priority', on: ['unavailable', 'overloaded', 'timeout'] }
        : undefined,
    }))
  }

  const save = async () => {
    const primary = modelName.trim()
    if (!primary) {
      setError('Give this model a name.')
      return
    }
    if (!recipe || recipe.decisions.length === 0) {
      setError('Choose a complete recipe.')
      return
    }
    const missing = recipe.decisions.find(
      (decision) => (assignments[decision.id]?.length ?? 0) === 0,
    )
    if (missing) {
      setError(`Choose at least one model for “${missing.name}”.`)
      return
    }
    for (const decision of recipe.decisions) {
      const selected = assignments[decision.id] ?? []
      const fallback = fallbacks[decision.id]
      if (!fallback && selected.some((assignment) => (assignment.priority ?? 0) !== 0)) {
        setError(`Keep every “${decision.name}” model at priority 0, or turn on fallback.`)
        return
      }
      if (fallback) {
        const tiers = [...new Set(selected.map((assignment) => assignment.priority ?? 0))].sort(
          (a, b) => a - b,
        )
        if (
          decision.dispatchCardinality !== 'single' ||
          tiers.length < 2 ||
          tiers.some((tier, index) => tier !== index)
        ) {
          setError(`Use consecutive priorities starting at 0 for “${decision.name}”.`)
          return
        }
      }
    }

    const publicAliases = [
      primary,
      ...aliases
        .split(/\r?\n/)
        .map((value) => value.trim())
        .filter(Boolean),
    ]
    const editedRule = {
      id: initialRule?.id,
      name: initialRule?.name ?? 'Default',
      matchers: initialRule?.matchers,
      recipeId,
      assignments: Object.fromEntries(
        recipe.decisions.map((decision) => [
          decision.id,
          {
            models: assignments[decision.id],
            ...(fallbacks[decision.id] ? { fallback: fallbacks[decision.id] } : {}),
          },
        ]),
      ),
    }
    const remainingRules = (entrypoint?.rules ?? []).slice(1).map(assignmentRuleWrite)
    setSaving(true)
    setError(null)
    try {
      await onSave({
        name: entrypoint?.name ?? primary,
        aliases: publicAliases,
        rules: [editedRule, ...remainingRules],
      })
      onClose()
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Model could not be saved.')
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
        aria-busy={saving}
        tabIndex={-1}
      >
        <header className={styles.header}>
          <div>
            <span>Mixture-of-Models</span>
            <h2 id={titleId}>{entrypoint ? 'Edit mixture' : 'Create a mixture'}</h2>
            <p>Choose a recipe. Complete every decision.</p>
          </div>
          <button type="button" onClick={onClose} disabled={saving} aria-label="Close">
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
                value={modelName}
                onChange={(event) => setModelName(event.target.value)}
                placeholder="my-company/auto"
                data-dialog-initial-focus
              />
            </label>
            <label>
              <span>Recipe</span>
              <select value={recipeId} onChange={(event) => chooseRecipe(event.target.value)}>
                {recipes.map((item) => (
                  <option key={item.id} value={item.id}>
                    {item.name}
                  </option>
                ))}
              </select>
            </label>
          </div>
          <section className={styles.decisionSection}>
            <div className={styles.sectionHeader}>
              <div>
                <span>Model assignments</span>
                <strong>{recipe?.decisions.length ?? 0} decisions</strong>
              </div>
              <p>{recipe?.description}</p>
            </div>
            <div className={styles.assignments}>
              {(recipe?.decisions ?? []).map((decision, index) => {
                const selected = assignments[decision.id] ?? []
                const fallbackEnabled = Boolean(fallbacks[decision.id])
                return (
                  <details
                    key={decision.id}
                    className={styles.assignment}
                    open={index === 0 || (recipe?.decisions.length ?? 0) <= 3}
                  >
                    <summary>
                      <span>
                        <strong>{decision.name}</strong>
                        <small>
                          {decision.dispatchCardinality === 'single'
                            ? 'Single dispatch'
                            : 'Multi-model dispatch'}
                        </small>
                      </span>
                      <span
                        className={`${styles.selectionCount} ${selected.length === 0 ? styles.missingCount : ''}`}
                      >
                        {selected.length === 0 ? 'Choose models' : `${selected.length} selected`}
                      </span>
                    </summary>
                    <div className={styles.assignmentBody}>
                      <div className={styles.modelPicker}>
                        {orderedModels.map((model) => (
                          <label
                            key={model.id}
                            className={
                              selected.some((value) => value.modelId === model.id)
                                ? styles.selectedModel
                                : ''
                            }
                          >
                            <input
                              type="checkbox"
                              checked={selected.some((value) => value.modelId === model.id)}
                              onChange={() => toggleModel(decision.id, model.id)}
                            />
                            <ProductIcon name="check" aria-hidden="true" />
                            <code>{model.name}</code>
                          </label>
                        ))}
                      </div>
                      {selected.length > 0 ? (
                        <div className={styles.pickerToolbar}>
                          {selected.map((value) => (
                            <label key={value.modelId}>
                              <code>
                                {models.find((model) => model.id === value.modelId)?.name ??
                                  value.modelId}
                              </code>
                              <input
                                type="number"
                                min={0}
                                max={31}
                                aria-label={`Priority for ${value.modelId}`}
                                value={value.priority ?? 0}
                                disabled={!fallbackEnabled}
                                onChange={(event) =>
                                  setAssignments((current) => ({
                                    ...current,
                                    [decision.id]: (current[decision.id] ?? []).map((item) =>
                                      item.modelId === value.modelId
                                        ? { ...item, priority: Number(event.target.value) }
                                        : item,
                                    ),
                                  }))
                                }
                              />
                            </label>
                          ))}
                        </div>
                      ) : null}
                      {decision.dispatchCardinality === 'single' && selected.length > 1 ? (
                        <label>
                          <input
                            type="checkbox"
                            checked={fallbackEnabled}
                            onChange={(event) => setFallback(decision.id, event.target.checked)}
                          />{' '}
                          Try the next priority when this model can’t start
                        </label>
                      ) : null}
                    </div>
                  </details>
                )
              })}
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
            <ProductIcon name="close" />
            Cancel
          </button>
          <button
            type="button"
            className={styles.save}
            onClick={() => void save()}
            disabled={saving}
          >
            <ProductIcon name={entrypoint ? 'check' : 'plus'} />
            {saving ? 'Saving…' : entrypoint ? 'Save' : 'Create mixture'}
          </button>
        </footer>
      </div>
    </div>
  )
}
