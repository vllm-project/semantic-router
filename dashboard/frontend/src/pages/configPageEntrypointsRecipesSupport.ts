import type {
  ConfigData,
  DecisionConfig,
  EntrypointConfig,
  NormalizedModel,
  RecipeConfig,
} from './configPageSupport'

export const DEFAULT_RECIPE_NAME = 'default'

export interface EntrypointFormState {
  modelNames: string
  recipe: string
}

export interface RecipeFormState {
  name: string
  description: string
  decisions: DecisionConfig[]
}

export function normalizeEntrypointModelNames(value: string | string[]): string[] {
  const values = Array.isArray(value) ? value : value.split(/[\n,]/)
  return [...new Set(values.map((item) => item.trim()).filter(Boolean))]
}

export function getRecipeNames(config: ConfigData | null): string[] {
  return [
    ...new Set([
      DEFAULT_RECIPE_NAME,
      ...(config?.recipes ?? []).map((recipe) => recipe.name).filter(Boolean),
    ]),
  ]
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
      description: 'Top-level routing profile used by vllm-sr/auto.',
      routing: {
        signals: config?.routing?.signals ?? config?.signals,
        projections: config?.routing?.projections ?? config?.projections,
        decisions: config?.routing?.decisions ?? config?.decisions ?? [],
      },
    }
  }
  return (config?.recipes ?? []).find((recipe) => recipe.name === recipeName) ?? null
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
  const modelNames = normalizeEntrypointModelNames(form.modelNames)
  if (modelNames.length === 0) {
    throw new Error('Add at least one request-facing model name.')
  }

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
  const autoNames = new Set(
    (
      config.global?.router?.auto_model_names ?? [
        'vllm-sr/auto',
        'auto',
        config.global?.router?.auto_model_name?.trim() || 'MoM',
      ]
    )
      .map((name) => name.trim())
      .filter(Boolean),
  )
  const directNames = new Set<string>()
  const looper = (config.global?.integrations?.looper ?? {}) as Record<string, unknown>
  for (const [family, defaultName] of [
    ['remom', 'vllm-sr/remom'],
    ['fusion', 'vllm-sr/fusion'],
    ['flow', 'vllm-sr/flow'],
  ] as const) {
    const familyConfig = looper[family]
    const modelNames =
      familyConfig && typeof familyConfig === 'object'
        ? (familyConfig as Record<string, unknown>).model_names
        : undefined
    const customNames = Array.isArray(modelNames)
      ? modelNames.filter(
          (name): name is string => typeof name === 'string' && Boolean(name.trim()),
        )
      : []
    if (customNames.length > 0) {
      customNames.forEach((name) => directNames.add(name.trim()))
      continue
    }
    directNames.add(defaultName)
  }

  for (const modelName of modelNames) {
    if (claimedNames.has(modelName)) {
      throw new Error(`Entrypoint model "${modelName}" is already mapped.`)
    }
    if (physicalNames.has(modelName) || loraNames.has(modelName)) {
      throw new Error(`Entrypoint model "${modelName}" collides with a configured model or LoRA.`)
    }
    if (autoNames.has(modelName)) {
      throw new Error(`Entrypoint model "${modelName}" is reserved as an auto-model alias.`)
    }
    if (directNames.has(modelName)) {
      throw new Error(`Entrypoint model "${modelName}" is reserved for direct router dispatch.`)
    }
  }

  return { model_names: modelNames, recipe }
}

export function validateRecipeForm(
  form: RecipeFormState,
  config: ConfigData,
  models: NormalizedModel[],
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

  const knownModels = new Set(
    models.flatMap((model) => [model.name, ...(model.loras ?? []).map((adapter) => adapter.name)]),
  )
  const decisionNames = new Set<string>()
  for (const decision of config.routing?.decisions ?? config.decisions ?? []) {
    decisionNames.add(decision.name)
  }
  for (const recipe of config.recipes ?? []) {
    if (recipe.name === originalName) continue
    for (const decision of recipe.routing.decisions ?? []) {
      decisionNames.add(decision.name)
    }
  }

  const decisions = (form.decisions ?? []).map((decision, index) => {
    const decisionName = decision.name?.trim()
    if (!decisionName) throw new Error(`Decision #${index + 1} needs a name.`)
    if (decisionNames.has(decisionName)) {
      throw new Error(`Decision name "${decisionName}" is already used by another recipe.`)
    }
    decisionNames.add(decisionName)

    const modelRefs = (decision.modelRefs ?? []).map((reference, refIndex) => {
      const model = reference.model?.trim()
      if (!model) {
        throw new Error(`Decision "${decisionName}" model reference #${refIndex + 1} is empty.`)
      }
      if (!knownModels.has(model)) {
        throw new Error(`Decision "${decisionName}" references unknown model "${model}".`)
      }
      return { ...reference, model }
    })
    return {
      ...decision,
      name: decisionName,
      description: decision.description?.trim() ?? '',
      priority: Number.isFinite(decision.priority) ? decision.priority : 0,
      rules: decision.rules ?? { operator: 'AND', conditions: [] },
      modelRefs,
    }
  })

  const existingRouting = (config.recipes ?? []).find(
    (recipe) => recipe.name === originalName,
  )?.routing

  return {
    name,
    description: form.description.trim() || undefined,
    routing: {
      ...existingRouting,
      decisions,
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
