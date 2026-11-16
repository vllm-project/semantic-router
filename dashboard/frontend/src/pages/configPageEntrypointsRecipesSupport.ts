import type {
  ConfigData,
  DecisionConfig,
  EntrypointConfig,
  NormalizedModel,
  RecipeConfig,
  RoutingStrategy,
} from './configPageSupport'
import { DEFAULT_ROUTING_STRATEGY } from './configPageSupport'

export const DEFAULT_RECIPE_NAME = 'default'

export interface EntrypointFormState {
  modelNames: string
  reviewModel?: string
  memoryModel?: string
  recipe: string
}

/** The bindings that send an agent's reviews or memory calls to another entrypoint. */
const DELEGATES = [
  { key: 'review_model', field: 'reviewModel', label: 'Reviews go to', noun: 'review' },
  { key: 'memory_model', field: 'memoryModel', label: 'Memories go to', noun: 'memory' },
] as const

/** Whether an entrypoint sends reviews or memory calls on, so it cannot take them. */
export const delegatesCalls = (entry: EntrypointConfig) => DELEGATES.some(({ key }) => entry[key])

export interface RecipeFormState {
  name: string
  description: string
  strategy: RoutingStrategy
}

export function normalizeRecipeStrategy(value: unknown): RoutingStrategy {
  return value === 'confidence' ? 'confidence' : DEFAULT_ROUTING_STRATEGY
}

export function normalizeEntrypointModelNames(value: string | string[]): string[] {
  const values = Array.isArray(value) ? value : value.split(/[\n,]/)
  return [...new Set(values.map((item) => item.trim()).filter(Boolean))]
}

export function requireEntrypointModelNames(value: string | string[]): string[] {
  const modelNames = normalizeEntrypointModelNames(value)
  if (modelNames.length === 0) {
    throw new Error('Add at least one request-facing model name.')
  }
  return modelNames
}

export function getRecipeNames(config: ConfigData | null): string[] {
  return [
    ...new Set([
      DEFAULT_RECIPE_NAME,
      ...(config?.recipes ?? []).map((recipe) => recipe.name).filter(Boolean),
    ]),
  ]
}

export function getPublishableRecipeNames(config: ConfigData | null): string[] {
  return getRecipeNames(config).filter(
    (name) => (getRecipeByName(config, name)?.routing.decisions?.length ?? 0) > 0,
  )
}

export function getRecipeByName(
  config: ConfigData | null,
  recipeName: string,
): RecipeConfig | null {
  if (recipeName === DEFAULT_RECIPE_NAME) {
    const explicitDefault = (config?.recipes ?? []).find(
      (recipe) => recipe.name === DEFAULT_RECIPE_NAME,
    )
    if (explicitDefault) return explicitDefault
    return {
      name: DEFAULT_RECIPE_NAME,
      description: 'Default routing profile. Entrypoints define its public model names.',
      routing: {
        signals: config?.routing?.signals ?? config?.signals,
        projections: config?.routing?.projections ?? config?.projections,
        decisions: config?.routing?.decisions ?? config?.decisions ?? [],
        strategy: normalizeRecipeStrategy(
          config?.routing?.strategy ?? config?.global?.router?.strategy,
        ),
      },
    }
  }
  return (config?.recipes ?? []).find((recipe) => recipe.name === recipeName) ?? null
}

export function listRecipeProfiles(config: ConfigData | null): RecipeConfig[] {
  return getRecipeNames(config)
    .map((name) => getRecipeByName(config, name))
    .filter((recipe): recipe is RecipeConfig => recipe !== null)
}

export function isSyntheticDefaultRecipe(config: ConfigData, recipe: RecipeConfig): boolean {
  return (
    recipe.name === DEFAULT_RECIPE_NAME &&
    !(config.recipes ?? []).some((candidate) => candidate.name === DEFAULT_RECIPE_NAME)
  )
}

export function collectRecipeTargetModels(recipe: RecipeConfig | null): string[] {
  if (!recipe) return []
  const models = new Set<string>()
  for (const decision of recipe.routing.decisions ?? []) {
    for (const modelName of collectDecisionTargetModels(decision)) {
      models.add(modelName)
    }
  }
  return [...models]
}

export function collectDecisionTargetModels(decision: DecisionConfig): string[] {
  const models = new Set<string>()
  for (const reference of decision.modelRefs ?? []) {
    const modelName = reference.model?.trim()
    if (modelName) models.add(modelName)
  }
  collectAlgorithmModelReferences(decision.algorithm, models)
  collectAlgorithmModelReferences(decision.candidateIterations, models)
  return [...models]
}

export function countRecipeEntrypoints(
  entrypoints: EntrypointConfig[],
  recipeName: string,
): number {
  return entrypoints
    .filter((entrypoint) => entrypoint.recipe === recipeName)
    .reduce((count, entrypoint) => count + entrypoint.model_names.length, 0)
}

export function validateEntrypointForm(
  form: EntrypointFormState,
  config: ConfigData,
  models: NormalizedModel[],
  originalIndex: number | null,
): EntrypointConfig {
  const modelNames = requireEntrypointModelNames(form.modelNames)

  const recipe = form.recipe.trim()
  if (!getRecipeNames(config).includes(recipe)) {
    throw new Error(`Recipe "${recipe || '(empty)'}" does not exist.`)
  }

  const claimedNames = new Set(
    (config.entrypoints ?? []).flatMap((entrypoint, index) =>
      index === originalIndex ? [] : entrypoint.model_names,
    ),
  )
  const physicalNames = new Set(models.map((model) => model.name))
  const loraNames = new Set(
    (config.routing?.modelCards ?? []).flatMap((model) =>
      (model.loras ?? []).map((lora) => lora.name),
    ),
  )
  const defaultDeclared = (config.entrypoints ?? []).some(
    (entrypoint, index) => index !== originalIndex && entrypoint.recipe === DEFAULT_RECIPE_NAME,
  )
  if (!defaultDeclared && recipe !== DEFAULT_RECIPE_NAME) claimedNames.add('vllm-sr/auto')
  for (const modelName of modelNames) {
    if (claimedNames.has(modelName)) {
      throw new Error(`Entrypoint model "${modelName}" is already mapped.`)
    }
    if (physicalNames.has(modelName) || loraNames.has(modelName)) {
      throw new Error(`Entrypoint model "${modelName}" collides with a configured model or LoRA.`)
    }
  }

  const api = originalIndex === null ? 'chat' : (config.entrypoints?.[originalIndex]?.api ?? 'chat')
  const delegates = DELEGATES.map((delegate) => ({
    ...delegate,
    name: form[delegate.field]?.trim(),
  }))
  for (const { label, noun, name } of delegates) {
    if (!name) continue
    if (modelNames.includes(name)) {
      throw new Error(`${label} must name another entrypoint.`)
    }
    const target = (config.entrypoints ?? []).find(
      (entry, index) =>
        index !== originalIndex &&
        (entry.api ?? 'chat') === api &&
        entry.model_names.includes(name),
    )
    if (!target) throw new Error(`The ${noun} entrypoint "${name}" does not exist.`)
    if (delegatesCalls(target)) {
      throw new Error(`A ${noun} entrypoint cannot have its own review_model or memory_model.`)
    }
  }
  const originalNames =
    originalIndex === null ? [] : (config.entrypoints?.[originalIndex]?.model_names ?? [])
  for (const entry of config.entrypoints ?? []) {
    if ((entry.api ?? 'chat') !== api) continue
    for (const { key, noun } of DELEGATES) {
      const used = entry[key]
      if (!used || !originalNames.includes(used)) continue
      if (!modelNames.includes(used)) {
        throw new Error(`Keep ${noun} model name "${used}" while another entrypoint uses it.`)
      }
      if (delegates.some(({ name }) => name)) {
        throw new Error(
          `An entrypoint used as a ${noun} model cannot have its own review_model or memory_model.`,
        )
      }
    }
  }
  const result: EntrypointConfig = {
    ...(originalIndex === null ? {} : config.entrypoints?.[originalIndex]),
    model_names: modelNames,
    recipe,
  }
  for (const { key, name } of delegates) {
    delete result[key]
    if (name) result[key] = name
  }
  return result
}

export function validateRecipeForm(
  form: RecipeFormState,
  config: ConfigData,
  _models: NormalizedModel[],
  originalName: string | null,
): RecipeConfig {
  const name = form.name.trim()
  if (!name) throw new Error('Recipe name is required.')
  if (originalName === DEFAULT_RECIPE_NAME && name !== DEFAULT_RECIPE_NAME) {
    throw new Error('The explicit default recipe cannot be renamed.')
  }
  if (name === DEFAULT_RECIPE_NAME && originalName !== DEFAULT_RECIPE_NAME) {
    throw new Error('The default recipe is owned by the top-level routing profile.')
  }
  if (
    (config.recipes ?? []).some((recipe) => recipe.name === name && recipe.name !== originalName)
  ) {
    throw new Error(`Recipe "${name}" already exists.`)
  }

  const existingRouting = (config.recipes ?? []).find(
    (recipe) => recipe.name === originalName,
  )?.routing

  return {
    name,
    description: form.description.trim() || undefined,
    routing: {
      ...existingRouting,
      strategy: form.strategy,
      signals: existingRouting?.signals ?? {},
      projections: existingRouting?.projections ?? {},
      decisions: existingRouting?.decisions ?? [],
    },
  }
}

export function getRecipeDeleteBlocker(config: ConfigData, recipeName: string): string | null {
  if (recipeName === DEFAULT_RECIPE_NAME) {
    return 'The explicit default recipe cannot be deleted from this surface.'
  }
  const aliases = countRecipeEntrypoints(config.entrypoints ?? [], recipeName)
  if (aliases > 0) {
    return `Move or delete ${aliases} entrypoint ${aliases === 1 ? 'model' : 'models'} before deleting this recipe.`
  }
  return null
}

function collectAlgorithmModelReferences(
  value: unknown,
  references: Set<string>,
  fieldName = '',
): void {
  if (typeof value === 'string') {
    if (fieldName === 'model' || fieldName.endsWith('_model')) {
      const modelName = value.trim()
      if (modelName) references.add(modelName)
    }
    return
  }
  if (Array.isArray(value)) {
    if (fieldName === 'models' || fieldName === 'model_names' || fieldName.endsWith('_models')) {
      for (const item of value) {
        if (typeof item === 'string' && item.trim()) references.add(item.trim())
        else collectAlgorithmModelReferences(item, references)
      }
      return
    }
    for (const item of value) collectAlgorithmModelReferences(item, references)
    return
  }
  if (!value || typeof value !== 'object') return
  for (const [key, nested] of Object.entries(value)) {
    collectAlgorithmModelReferences(nested, references, key)
  }
}
