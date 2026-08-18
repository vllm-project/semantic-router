import type { RecipeProbeMessage, RecipeProbeRunPlan } from './recipe'

export interface PlaygroundInvocation {
  version: 1
  intent: 'run' | 'edit'
  source: 'recipe-probe'
  probeId: string
  recipeDigest: string
  editable: boolean
  model?: string
  messages: RecipeProbeMessage[]
  tools?: unknown[]
  request: Record<string, unknown>
}

export interface PlaygroundInvocationLocationState {
  playgroundInvocation: PlaygroundInvocation
}

export function createProbePlaygroundInvocation(
  intent: PlaygroundInvocation['intent'],
  plan: RecipeProbeRunPlan,
): PlaygroundInvocation {
  return {
    version: 1,
    intent,
    source: 'recipe-probe',
    probeId: plan.probe_id,
    recipeDigest: plan.recipe_digest,
    editable: plan.editable,
    ...(plan.model ? { model: plan.model } : {}),
    messages: plan.messages,
    ...(plan.tools ? { tools: plan.tools } : {}),
    request: plan.request,
  }
}

export function isPlaygroundInvocation(value: unknown): value is PlaygroundInvocation {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return false
  const candidate = value as Partial<PlaygroundInvocation>
  return (
    candidate.version === 1 &&
    (candidate.intent === 'run' || candidate.intent === 'edit') &&
    candidate.source === 'recipe-probe' &&
    typeof candidate.probeId === 'string' &&
    typeof candidate.recipeDigest === 'string' &&
    typeof candidate.editable === 'boolean' &&
    Array.isArray(candidate.messages) &&
    Boolean(candidate.request) &&
    typeof candidate.request === 'object' &&
    !Array.isArray(candidate.request)
  )
}
