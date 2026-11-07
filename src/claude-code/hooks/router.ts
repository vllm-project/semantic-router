import { atom, read, update } from 'claude-code'
import type { AgentSpawnInput, EngineInterface, Hook, TurnCompleteInput, TurnStepInput } from 'claude-code'

import type { Model, Spend, Steps, Tiers } from '../types'
import { classify, numberAt, offlineOf, type Reply } from './lib/classify'
import { effortFor, type Picked, pickEffort, raised, type StepsLines } from './lib/effort'
import type { Settings } from './lib/settings'
import { costOf } from './lib/prices'
import { readDecisions, readEscalation, record, stamp, updateDecision } from './lib/state'

// Past this the spawn goes on as it would have without the plugin (fail open).
const SPAWN_TIMEOUT_MS = 1500
// Only these subagent types are routed: others (Explore, plugin agents) keep the model their definition picks.
const ROUTED_TYPES = new Set(['general-purpose'])

// The Router's built-in `complexity` task, asked word for word.
const COMPLEXITY = {
  type: 'score',
  instructions: 'How difficult is this request to solve correctly?',
  criteria: [
    'straightforward retrieval or formatting',
    'several steps of reasoning',
    'advanced reasoning or specialized expertise',
  ],
}

// The `steps` question: how many reasoning steps the task needs, asked in the same request as `complexity`.
const STEPS = {
  type: 'score',
  instructions: 'How many reasoning steps does this task need?',
  criteria: ['one or two steps', 'several steps that build on each other', 'many steps or long careful reasoning'],
}

// `floor`: the effort never drops below the one the subagent would have had.
type Route = { model: Model; floor: boolean }

// By the expected difficulty level (the Score answer's `score`, 0 to 2).
function routeOf(score: number, { easyBelow, hardFrom }: Tiers): Route {
  if (score < easyBelow) return { model: 'haiku', floor: false }
  if (score < hardFrom) return { model: 'sonnet', floor: false }
  return { model: 'opus', floor: true }
}

const RANK: Record<Model, number> = { haiku: 0, sonnet: 1, opus: 2 }

// The model a subagent has when the router leaves it alone: the session's, by its id. A model above Opus
// (Fable, say) ranks above every route, so `model` is undefined: the router never names it.
function defaultOf(parentModel: string): { model: Model | undefined; rank: number } {
  const model = (['haiku', 'sonnet', 'opus'] as const).find(name => parentModel.includes(name))
  return model === undefined ? { model, rank: 3 } : { model, rank: RANK[model] }
}

// After a pushback the next subagent climbs one step: never below the session's model, one effort level above
// what it would have had, and only from Haiku, whose effort is fixed, to the next model.
function escalatedRoute(alone: Model | undefined, fallback: ReturnType<typeof defaultOf>, steps: Steps | null, lines: StepsLines) {
  const base = alone !== undefined && RANK[alone] > fallback.rank ? alone : fallback.model
  const model = base === 'haiku' ? 'sonnet' : base
  const picked: Picked = { effort: raised(pickEffort(model ?? 'opus', steps, lines).effort), from: 'escalated' }
  return { model, picked }
}

// A usable answer, or none: a missing or malformed steps answer is the same as no answer.
function scoreOf(answer: unknown): Steps | undefined {
  const score = numberAt(answer, 'score')
  const confidence = numberAt(answer, 'confidence')
  return score === undefined || confidence === undefined ? undefined : { score, confidence }
}

// Only a failed complexity answer fails the parse; a failed steps answer is `null`.
function parseAnswers({ answers }: Reply) {
  const complexity = scoreOf(answers.complexity)
  return complexity === undefined ? undefined : { complexity, steps: scoreOf(answers.steps) ?? null }
}

// The route each routed subagent runs on, by its agent id.
const routes = new Map<string, { picked: Picked; floor: boolean }>()

export async function routeSpawn(
  $: EngineInterface,
  e: AgentSpawnInput,
  next: Parameters<Hook<'agent.spawn'>>[2],
  settings: Settings,
) {
  if (!settings.apps.router || e.fork || e.isTeammate || e.model !== undefined || !ROUTED_TYPES.has(e.subagentType)) {
    return next(e)
  }
  const { router } = settings.thresholds
  const lines = { low: router.stepsLow, high: router.stepsHigh, minConfidence: router.stepsMinConfidence }
  const tiers = { easyBelow: router.easyBelow, hardFrom: router.hardFrom }
  const fallback = defaultOf(e.parentModel)
  const isEscalated = (await readEscalation($)) !== null
  const asked = await classify($, settings.runtime, e.prompt, { complexity: COMPLEXITY, steps: STEPS }, SPAWN_TIMEOUT_MS, parseAnswers)
  const at = await stamp($)
  const anchor = { id: e.tool_use_id, toolUseId: e.tool_use_id, at }
  if (!asked.ok) {
    const why = isEscalated ? `${asked.why} · escalated, so one step up anyway` : asked.why
    await record($, settings.log, { kind: 'unavailable', app: 'router', ...anchor, subject: e.description, why, ...offlineOf(asked) })
    if (!isEscalated) return next(e)
    const up = escalatedRoute(undefined, fallback, null, lines)
    const result = await next(up.model === undefined ? e : { ...e, model: up.model })
    if (result.agentId) routes.set(result.agentId, { picked: up.picked, floor: true })
    return result
  }
  const { complexity: { score, confidence }, steps } = asked.value
  const facts = { task: e.description, level: score, confidence, tiers, latencyMs: asked.latencyMs }
  const alone = routeOf(score, tiers)
  const isSure = confidence >= router.minConfidence
  // A smaller model than the session's costs the most when the router is wrong, so it needs more certainty.
  const isSmaller = RANK[alone.model] < fallback.rank
  const smallerLine = settings.routerGoal === 'save' ? router.minConfidence : Math.max(router.minConfidence, router.downgradeConfidence)
  const reason = !isSure ? 'unsure' : !isSmaller ? undefined : settings.routerGoal === 'quality' ? 'quality' : confidence < smallerLine ? 'downgrade' : undefined
  if (reason !== undefined && !isEscalated) {
    const result = await next(e)
    await record($, settings.log, {
      kind: 'keep',
      ...anchor,
      id: result.agentId ?? e.tool_use_id,
      model: result.model,
      minConfidence: reason === 'downgrade' ? smallerLine : router.minConfidence,
      reason,
      ...facts,
    })
    return result
  }
  const up = isEscalated ? escalatedRoute(isSure ? alone.model : undefined, fallback, steps, lines) : undefined
  const model = up ? up.model : alone.model
  const floor = up ? true : alone.floor
  const picked = up ? up.picked : pickEffort(alone.model, steps, lines)
  const result = await next(model === undefined ? e : { ...e, model })
  if (result.agentId) routes.set(result.agentId, { picked, floor })
  await record($, settings.log, {
    kind: 'route',
    ...anchor,
    id: result.agentId ?? e.tool_use_id,
    agentId: result.agentId,
    model: model ?? result.model,
    picked: picked.effort,
    steps,
    effortFrom: picked.from,
    ...facts,
    ...(isEscalated ? { escalated: { alone: isSure ? alone.model : 'default' } } : {}),
    sessionModel: e.parentModel,
  })
  return result
}

const NO_SPEND: Spend = { actual: 0, baseline: 0, turns: 0 }
const spend = atom({ plugin: 'vsr', key: 'spend' } as const, NO_SPEND)

// Read while drawing: what routed subagents cost, and what the same tokens would have cost on the session's model.
export async function readSpend($: EngineInterface): Promise<Spend> {
  return read($, spend)
}

// A routed subagent's turn, priced at list price on the model that answered and on the session's model.
export async function countSpend($: EngineInterface, e: TurnCompleteInput) {
  if (e.agentId === undefined || e.usage === undefined) return
  const route = (await readDecisions($)).find(d => d.kind === 'route' && d.agentId === e.agentId)
  if (route?.kind !== 'route' || route.sessionModel === undefined) return
  const actual = costOf(e.usage.model, e.usage)
  const baseline = costOf(route.sessionModel, e.usage)
  if (actual === undefined || baseline === undefined) return
  await update($, spend, now => ({ actual: now.actual + actual, baseline: now.baseline + baseline, turns: now.turns + 1 }))
}

export async function* routeStep(
  $: EngineInterface,
  e: TurnStepInput,
  next: Parameters<Hook<'turn.step'>>[2],
  settings: Settings,
) {
  const agentId = e.agentId
  const route = agentId ? routes.get(agentId) : undefined
  if (route === undefined || typeof e.effort !== 'string') return yield* next(e)
  const { effort, from } = effortFor(route.picked, route.floor, e.effort)
  if (e.index === 0) {
    await updateDecision(
      $,
      settings.log,
      d => d.kind === 'route' && d.agentId === agentId,
      d => ({ ...d, effort, effortFrom: from }),
    )
  }
  return yield* next({ ...e, effort })
}
