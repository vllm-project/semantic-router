import type {
  ConfigData,
  EntrypointConfig,
  ProviderModelConfig,
  RecipeConfig,
  RecipeRoutingConfig,
  RoutingConfig,
  RoutingModelCard,
} from './configPageSupport'

// The Router's configuration (v0.5) keeps routing in flat recipes and model
// facts on providers.models[]. The config manager edits a default routing
// profile, named recipes and model cards; these two functions are the only
// place that converts between the two.

export const CANONICAL_CONFIG_VERSION = 'v0.5'
export const DEFAULT_RECIPE_NAME = 'default'
export const DEFAULT_ENTRYPOINT_MODEL = 'vllm-sr/auto'

// The keys of the facts a providers.models[] entry states.
const MODEL_FACT_KEYS = [
  'display_name',
  'publisher',
  'presentation',
  'distribution',
  'family',
  'revision',
  'released_at',
  'knowledge_cutoff',
  'lifecycle',
  'param_size',
  'context_window_size',
  'max_output_tokens',
  'description',
  'capabilities',
  'loras',
  'modalities',
  'tags',
] as const satisfies ReadonlyArray<keyof RoutingModelCard>

type FlatRecipe = RecipeRoutingConfig & RoutingConfig & { name: string; description?: string }
type CanonicalEntrypoint = EntrypointConfig & { api?: string; default?: boolean }
type MutableRecord = Record<string, unknown>

const clone = <T>(value: T): T => JSON.parse(JSON.stringify(value)) as T

const isFlatLayout = (config: ConfigData): boolean =>
  !config.routing &&
  (config.recipes ?? []).every((recipe) => !(recipe as Partial<RecipeConfig>).routing)

const isChatEntrypoint = (entrypoint: CanonicalEntrypoint) =>
  !entrypoint.api || entrypoint.api === 'chat'

function defaultRecipeName(recipes: FlatRecipe[], entrypoints: CanonicalEntrypoint[]): string {
  if (recipes.length === 1) return recipes[0].name
  const marked = entrypoints.find((entrypoint) => entrypoint.default)
  return marked?.recipe ?? DEFAULT_RECIPE_NAME
}

function profileOf(recipe: FlatRecipe | undefined): RoutingConfig {
  if (!recipe) return {}
  const profile: MutableRecord = { ...clone(recipe) }
  delete profile.name
  delete profile.description
  return profile as RoutingConfig
}

function takeModelCards(providers: ProviderModelConfig[]): RoutingModelCard[] {
  const cards: RoutingModelCard[] = []
  for (const model of providers) {
    const record = model as unknown as MutableRecord
    const card: MutableRecord = { name: model.name }
    for (const key of MODEL_FACT_KEYS) {
      if (record[key] !== undefined) {
        card[key] = record[key]
        delete record[key]
      }
    }
    if (Object.keys(card).length > 1) cards.push(card as unknown as RoutingModelCard)
  }
  return cards
}

/** The config manager's view of a v0.5 configuration. */
export function fromCanonicalLayout(config: ConfigData): ConfigData {
  const next = clone(config)
  const recipes = (next.recipes ?? []) as unknown as FlatRecipe[]
  const entrypoints = (next.entrypoints ?? []) as CanonicalEntrypoint[]
  const name = defaultRecipeName(recipes, entrypoints)
  const defaultRecipe = recipes.find((recipe) => recipe.name === name)

  next.routing = profileOf(defaultRecipe)
  const cards = takeModelCards(next.providers?.models ?? [])
  if (cards.length > 0) next.routing.modelCards = cards
  next.default_recipe = { name, description: defaultRecipe?.description }
  next.recipes = recipes
    .filter((recipe) => recipe.name !== name)
    .map((recipe) => ({
      name: recipe.name,
      ...(recipe.description ? { description: recipe.description } : {}),
      routing: profileOf(recipe),
    }))
  const named = entrypoints
    .filter(
      (entrypoint) =>
        !(
          entrypoint.default &&
          entrypoint.recipe === name &&
          entrypoint.model_names.length === 1 &&
          entrypoint.model_names[0] === DEFAULT_ENTRYPOINT_MODEL
        ),
    )
    .map((entrypoint) => {
      const view: CanonicalEntrypoint = {
        ...entrypoint,
        recipe: entrypoint.recipe === name ? DEFAULT_RECIPE_NAME : entrypoint.recipe,
      }
      delete view.default
      return view
    })
  if (named.length > 0) next.entrypoints = named
  else delete next.entrypoints
  return next
}

function attachModelFacts(next: ConfigData, cards: RoutingModelCard[]) {
  if (cards.length === 0) return
  if (!next.providers) next.providers = { models: [] }
  const models = next.providers.models
  for (const card of cards) {
    let model = models.find((candidate) => candidate.name === card.name)
    if (!model) {
      model = models.find((candidate) => candidate.catalog?.trim() === card.name)
    }
    if (!model) {
      model = { name: card.name }
      models.push(model)
    }
    const record = model as unknown as MutableRecord
    for (const key of MODEL_FACT_KEYS) {
      if (card[key] !== undefined && record[key] === undefined) record[key] = clone(card[key])
    }
  }
}

function markDefaultEntrypoint(entrypoints: CanonicalEntrypoint[], name: string) {
  if (entrypoints.some((entrypoint) => entrypoint.default)) return entrypoints
  const own = entrypoints.find(
    (entrypoint) => entrypoint.recipe === name && isChatEntrypoint(entrypoint),
  )
  if (own) {
    own.default = true
    return entrypoints
  }
  return [{ model_names: [DEFAULT_ENTRYPOINT_MODEL], recipe: name, default: true }, ...entrypoints]
}

/** The v0.5 configuration a config manager view describes. */
export function toCanonicalLayout(view: ConfigData): ConfigData {
  if (isFlatLayout(view) && !view.default_recipe) {
    const next = clone(view)
    next.version = CANONICAL_CONFIG_VERSION
    return next
  }
  const next = clone(view)
  const { modelCards = [], ...profile } = next.routing ?? {}
  const meta = next.default_recipe ?? { name: DEFAULT_RECIPE_NAME }
  const recipes: FlatRecipe[] = [
    {
      name: meta.name,
      ...(meta.description ? { description: meta.description } : {}),
      ...profile,
    },
    ...(next.recipes ?? []).map((recipe) => ({
      name: recipe.name,
      ...(recipe.description ? { description: recipe.description } : {}),
      ...(recipe.routing ?? profileOf(recipe as unknown as FlatRecipe)),
    })),
  ]
  attachModelFacts(next, modelCards)

  let entrypoints = (next.entrypoints ?? []).map((entrypoint) => ({
    ...entrypoint,
    recipe: entrypoint.recipe === DEFAULT_RECIPE_NAME ? meta.name : entrypoint.recipe,
  })) as CanonicalEntrypoint[]
  if (recipes.length > 1) entrypoints = markDefaultEntrypoint(entrypoints, meta.name)

  delete next.routing
  delete next.default_recipe
  next.version = CANONICAL_CONFIG_VERSION
  next.recipes = recipes as unknown as RecipeConfig[]
  if (entrypoints.length > 0) next.entrypoints = entrypoints
  else delete next.entrypoints
  return next
}

/** A fetched configuration in the shape the dashboard reads: a routing
 * profile, named recipes and model cards. */
export const withRoutingView = <T>(config: T): T =>
  fromCanonicalLayout(config as unknown as ConfigData) as unknown as T

/** The v0.5 configuration a routing view describes, to write it back. */
export const withCanonicalLayout = <T>(config: T): T =>
  toCanonicalLayout(config as unknown as ConfigData) as unknown as T
