import { useMemo, useState } from 'react'

import { type BuiltInRecipeCatalog, composeBuiltInRecipe } from './configPageBuiltInRecipes'
import {
  DEFAULT_RECIPE_NAME,
  delegatesCalls,
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

const firstDecisionName = (decisionNames: string[]) => decisionNames[0] ?? ''

export interface ConfigPageMixtureDialogProps {
  config: ConfigData
  models: NormalizedModel[]
  builtInRecipes?: BuiltInRecipeCatalog | null
  initialRecipe?: MixtureRecipeChoice
  entrypoint?: EntrypointConfig
  entrypointIndex?: number
  onClose: () => void
  onSave: (config: ConfigData) => Promise<void>
}

/** State of the public model editor: identity, recipe, delegates and model assignments. */
export function useMixtureEditor({
  config,
  models,
  builtInRecipes = null,
  initialRecipe,
  entrypoint,
  entrypointIndex,
  onClose,
  onSave,
}: ConfigPageMixtureDialogProps) {
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
  const [reviewModel, setReviewModel] = useState(entrypoint?.review_model ?? '')
  const [memoryModel, setMemoryModel] = useState(entrypoint?.memory_model ?? '')
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

  const publishConfigRecipe = (recipeName: string, publicNames: string[]): ConfigData => {
    const normalizedEntrypoint = validateEntrypointForm(
      { modelNames: publicNames.join('\n'), recipe: recipeName, reviewModel, memoryModel },
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

  // A name problem is shown as the user types; the full check runs again on save.
  const nameProblem = useMemo(() => {
    if (!primaryName.trim()) return null
    try {
      validateEntrypointForm(
        {
          modelNames: [primaryName, ...aliases].join('\n'),
          recipe: choice && choice.bundle === undefined ? choice.name : DEFAULT_RECIPE_NAME,
        },
        config,
        models,
        entrypointIndex ?? null,
      )
      return null
    } catch (cause) {
      return cause instanceof Error ? cause.message : null
    }
  }, [aliases, choice, config, entrypointIndex, models, primaryName])

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
  // Other entrypoints that send nothing on can take this one's reviews and memory calls.
  const delegateNames = (config.entrypoints ?? [])
    .filter(
      (candidate, index) =>
        index !== entrypointIndex &&
        !delegatesCalls(candidate) &&
        (candidate.api ?? 'chat') === (entrypoint?.api ?? 'chat'),
    )
    .flatMap((candidate) => candidate.model_names)
    .filter((name) => ![primaryName.trim(), ...aliases].includes(name))

  return {
    entrypoint,
    models,
    catalog,
    recipeNames,
    choice,
    description,
    decisions,
    chooseRecipe,
    primaryName,
    setPrimaryName,
    aliases,
    setAliases,
    reviewModel,
    setReviewModel,
    memoryModel,
    setMemoryModel,
    assignments,
    toggleModel,
    activeDecision,
    setActiveDecisionName,
    activeSelection,
    decisionSearch,
    setDecisionSearch,
    visibleDecisions,
    modelSearch,
    setModelSearch,
    showSelectedModels,
    setShowSelectedModels,
    modelOptions,
    visibleModelOptions,
    completedDecisionCount,
    delegateNames,
    nameProblem,
    saving,
    error,
    save,
  }
}

export type MixtureEditor = ReturnType<typeof useMixtureEditor>
