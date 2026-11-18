import type { RecipeActivationPlan, RecipePackageList, RecipePackageSummary } from '../types/recipe'
import type { StatusTone } from '../ui'
import {
  countProjectionsInProfile,
  countSignalsInProfile,
  effectiveChatEntrypoints,
  type RoutingProfileLike,
} from '../utils/routingScopes'
import type { BuiltInRecipe, BuiltInRecipeCatalog } from './configPageBuiltInRecipes'
import {
  DEFAULT_RECIPE_NAME,
  collectDecisionTargetModels,
  collectRecipeTargetModels,
  getRecipeByName,
  isSyntheticDefaultRecipe,
  listRecipeProfiles,
} from './configPageEntrypointsRecipesSupport'
import type {
  ConfigData,
  DecisionConfig,
  EntrypointConfig,
  RecipeConfig,
} from './configPageSupport'

export interface PublicModelCard {
  /** The first model name; it identifies the card in the URL. */
  key: string
  entrypoint: EntrypointConfig
  /** Index in config.entrypoints; -1 for the derived `vllm-sr/auto` default. */
  index: number
  api: 'chat' | 'systemone'
  /** The derived default entrypoint, shown but never written back. */
  derived: boolean
  recipe: RecipeConfig | null
  signals: number
  signalFamilies: number
  projections: number
  decisions: DecisionConfig[]
  targets: string[]
}

export type PublicModelResolution = 'resolving' | 'empty' | 'missing'

export function publicModelResolution(card: Pick<PublicModelCard, 'recipe' | 'decisions'>) {
  if (!card.recipe) return 'missing' as const
  return card.decisions.length ? ('resolving' as const) : ('empty' as const)
}

export function recipeDisplayName(config: ConfigData | null, recipeName: string): string {
  return recipeName === DEFAULT_RECIPE_NAME && config?.default_recipe?.name
    ? config.default_recipe.name
    : recipeName
}

export function sortedDecisions(recipe: RecipeConfig | null): DecisionConfig[] {
  return [...(recipe?.routing.decisions ?? [])].sort(
    (left, right) => (right.priority ?? 0) - (left.priority ?? 0),
  )
}

export function publicModelCards(config: ConfigData): PublicModelCard[] {
  const authored = config.entrypoints ?? []
  const chat = effectiveChatEntrypoints(config) as EntrypointConfig[]
  const systemone = authored.filter((entrypoint) => entrypoint.api === 'systemone')
  return [...chat, ...systemone].map((entrypoint) => {
    const recipe = getRecipeByName(config, entrypoint.recipe)
    const profile = recipe?.routing as RoutingProfileLike | undefined
    const signals = countSignalsInProfile(profile)
    return {
      key: entrypoint.model_names[0] ?? entrypoint.recipe,
      entrypoint,
      index: authored.indexOf(entrypoint),
      api: entrypoint.api === 'systemone' ? 'systemone' : 'chat',
      derived: authored.indexOf(entrypoint) < 0,
      recipe,
      signals: signals.total,
      signalFamilies: Object.keys(signals.byType).length,
      projections: countProjectionsInProfile(profile),
      decisions: sortedDecisions(recipe),
      targets: collectRecipeTargetModels(recipe),
    }
  })
}

export interface ConfigRecipeRow {
  kind: 'config'
  id: string
  name: string
  description?: string
  recipe: RecipeConfig
  synthetic: boolean
  publicModels: string[]
  decisions: number
  signals: number
  projections: number
  targets: string[]
  strategy: string
}

export interface PackageRecipeRow {
  kind: 'package'
  id: string
  name: string
  description?: string
  summary: RecipePackageSummary
}

export interface BuiltInRecipeRow {
  kind: 'builtin'
  id: string
  name: string
  description?: string
  bundle: string
  catalogVersion: string
  recipe: BuiltInRecipe
}

export type RecipeRow = ConfigRecipeRow | PackageRecipeRow | BuiltInRecipeRow

export const packageRowId = (digest: string) => `package:${digest}`
export const builtInRowId = (bundle: string, name: string) => `builtin:${bundle}/${name}`

export function recipeRows(
  config: ConfigData,
  packages: RecipePackageList | null,
  builtIns: BuiltInRecipeCatalog | null,
): RecipeRow[] {
  const entrypoints = config.entrypoints ?? []
  const configRows: ConfigRecipeRow[] = listRecipeProfiles(config).map((recipe) => {
    const profile = recipe.routing as RoutingProfileLike
    const publicModels = [
      ...new Set(
        (effectiveChatEntrypoints(config) as EntrypointConfig[])
          .concat(entrypoints.filter((entrypoint) => entrypoint.api === 'systemone'))
          .filter((entrypoint) => entrypoint.recipe === recipe.name)
          .flatMap((entrypoint) => entrypoint.model_names),
      ),
    ]
    return {
      kind: 'config',
      id: recipe.name,
      name: recipeDisplayName(config, recipe.name),
      description:
        recipe.name === DEFAULT_RECIPE_NAME
          ? (config.default_recipe?.description ?? recipe.description)
          : recipe.description,
      recipe,
      synthetic: isSyntheticDefaultRecipe(config, recipe),
      publicModels,
      decisions: recipe.routing.decisions?.length ?? 0,
      signals: countSignalsInProfile(profile).total,
      projections: countProjectionsInProfile(profile),
      targets: collectRecipeTargetModels(recipe),
      strategy: String(recipe.routing.strategy ?? 'priority'),
    }
  })
  const packageRows: PackageRecipeRow[] = (packages?.items ?? []).map((summary) => ({
    kind: 'package',
    id: packageRowId(summary.recipe_digest),
    name: summary.name || summary.id,
    description: summary.description,
    summary,
  }))
  const builtInRows: BuiltInRecipeRow[] = (builtIns?.bundles ?? []).flatMap((bundle) =>
    bundle.recipes.map((recipe) => ({
      kind: 'builtin' as const,
      id: builtInRowId(bundle.name, recipe.name),
      name: recipe.name,
      description: recipe.description,
      bundle: bundle.name,
      catalogVersion: builtIns?.catalog_version ?? '',
      recipe,
    })),
  )
  return [...configRows, ...packageRows, ...builtInRows]
}

export function recipeRowStatus(row: RecipeRow, pendingRestart: ReadonlySet<string>) {
  if (row.kind === 'config') return 'loaded' as const
  if (row.kind === 'builtin') return 'available' as const
  if (pendingRestart.has(row.summary.recipe_digest)) return 'pending' as const
  return row.summary.active ? ('active' as const) : ('inactive' as const)
}

export function filterRecipeRows(
  rows: RecipeRow[],
  search: string,
  sources: readonly string[],
  statuses: readonly string[],
  pendingRestart: ReadonlySet<string>,
): RecipeRow[] {
  const query = search.trim().toLocaleLowerCase()
  return rows.filter((row) => {
    if (sources.length && !sources.includes(row.kind)) return false
    if (statuses.length && !statuses.includes(recipeRowStatus(row, pendingRestart))) return false
    if (!query) return true
    const values = [
      row.name,
      row.description,
      ...(row.kind === 'config' ? [...row.publicModels, ...row.targets] : []),
      ...(row.kind === 'package' ? [row.summary.id, row.summary.version] : []),
      ...(row.kind === 'builtin' ? [row.bundle] : []),
    ]
    return values.some((value) => value?.toLocaleLowerCase().includes(query))
  })
}

/** The decisions of a recipe with the models each can send traffic to. */
export function decisionLanes(recipe: RecipeConfig | null) {
  return sortedDecisions(recipe).map((decision) => ({
    decision,
    targets: collectDecisionTargetModels(decision),
  }))
}

export const ENTRYPOINT_SHEET_VIEWS = ['overview', 'backends', 'connect', 'yaml'] as const
export type EntrypointSheetView = (typeof ENTRYPOINT_SHEET_VIEWS)[number]

export const apiLabel = (card: Pick<PublicModelCard, 'api'>) =>
  card.api === 'systemone' ? 'System One' : 'OpenAI'

export function usageExamples(origin: string, model: string) {
  return [
    {
      label: 'cURL',
      language: 'bash' as const,
      code: `curl ${origin}/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -d '{
    "model": "${model}",
    "messages": [{"role": "user", "content": "Hello"}],
    "stream": true
  }'`,
    },
    {
      label: 'Python',
      language: 'python' as const,
      code: `from openai import OpenAI

client = OpenAI(
    base_url="${origin}/v1",
    api_key="not-required",
)

stream = client.chat.completions.create(
    model="${model}",
    messages=[{"role": "user", "content": "Hello"}],
    stream=True,
)

for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="")`,
    },
    {
      label: 'JavaScript',
      language: 'ts' as const,
      code: `import OpenAI from "openai";

const client = new OpenAI({
  baseURL: "${origin}/v1",
  apiKey: "not-required",
});

const stream = await client.chat.completions.create({
  model: "${model}",
  messages: [{ role: "user", content: "Hello" }],
  stream: true,
});

for await (const chunk of stream) {
  process.stdout.write(chunk.choices[0]?.delta?.content ?? "");
}`,
    },
  ]
}

export interface PlanChange {
  kind: 'add' | 'remove' | 'info'
  text: string
}

function listenerKey(listener: { name: string; address: string; port: number }) {
  return `${listener.name} ${listener.address}:${listener.port}`
}

/** The plan as a list of visible changes: listeners, managed storage and effects. */
export function planChanges(plan: RecipeActivationPlan): PlanChange[] {
  const before = new Set(plan.listeners_before.map(listenerKey))
  const after = new Set(plan.listeners_after.map(listenerKey))
  const changes: PlanChange[] = [
    ...[...after]
      .filter((key) => !before.has(key))
      .map((key) => ({
        kind: 'add' as const,
        text: `Listener ${key}`,
      })),
    ...[...before]
      .filter((key) => !after.has(key))
      .map((key) => ({
        kind: 'remove' as const,
        text: `Listener ${key}`,
      })),
    ...plan.storage.add.map((name) => ({ kind: 'add' as const, text: `Managed storage ${name}` })),
    ...plan.storage.remove.map((name) => ({
      kind: 'remove' as const,
      text: `Managed storage ${name}`,
    })),
    ...plan.storage.repair.map((name) => ({
      kind: 'info' as const,
      text: `Repairs managed storage ${name}`,
    })),
    ...plan.effects.map((text) => ({ kind: 'info' as const, text })),
  ]
  const management = plan.management_after
  if (
    management &&
    plan.management_before &&
    (management.port !== plan.management_before.port ||
      management.bind_address !== plan.management_before.bind_address)
  ) {
    changes.push({
      kind: 'info',
      text: `Management API moves to ${management.bind_address}:${management.port}`,
    })
  }
  return changes
}

export const restartRequired = (plan: RecipeActivationPlan | null) =>
  plan?.mode === 'stack_recreation'

export const recipeStatusLabel: Record<
  ReturnType<typeof recipeRowStatus>,
  { tone: StatusTone; label: string }
> = {
  loaded: { tone: 'success', label: 'Loaded' },
  active: { tone: 'success', label: 'Active' },
  inactive: { tone: 'neutral', label: 'Inactive' },
  pending: { tone: 'warning', label: 'Restart pending' },
  available: { tone: 'neutral', label: 'Available' },
}
