import { atom, read, update } from 'claude-code'
import type { AgentSpawnInput, EngineInterface, Hook, TurnCompleteInput, TurnStepInput } from 'claude-code'

import type { Model, Spend, Steps, Tiers } from '../types'
import { classify, modelOf, numberAt, failureOf, QUICK_ANSWER_MS, type Reply } from './lib/classify'
import { effortFor, type Picked, pickEffort, raised, type StepsLines } from './lib/effort'
import type { Runtime, Settings } from './lib/settings'
import { costOf } from './lib/prices'
import { readEscalation } from './escalate'
import { record, updateDecision } from './lib/state'
import { stamp } from './lib/env'

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

// A runtime model that answers no `score` question, such as vela3-fast: both router questions would fail on every
// subagent, so the router steps aside for the session. Set once, at session start.
const noScore = atom({ plugin: 'vsr', key: 'noScoreModel' } as const, '')

// Reads which questions the runtime's model answers and, when it answers no `score` question, says once why the
// router keeps every subagent on its default model.
export async function checkRouterModel($: EngineInterface, runtime: Runtime) {
  const model = await modelOf($, runtime)
  if (model === undefined || model.types.includes('score')) return
  await update($, noScore, () => model.id)
  $.ui.toast(
    `vSR router: ${model.id} answers no score questions, so subagents keep their default model · vela3-lite does · /vsr:setup`,
  )
}

// `floor`: the effort never drops below the one the subagent would have had.
type Route = { model: Model; floor: boolean }

// By the expected difficulty level (the Score answer's `score`, 0 to 2).
function routeOf(score: number, { easyBelow, hardFrom }: Tiers): Route {
  if (score < easyBelow) return { model: 'haiku', floor: false }
  if (score < hardFrom) return { model: 'sonnet', floor: false }
  return { model: 'opus', floor: true }
}

// The models a subagent can be routed to, cheapest first.
const MODELS: readonly Model[] = ['haiku', 'sonnet', 'opus']
const RANK = Object.fromEntries(MODELS.map((model, rank) => [model, rank])) as Record<Model, number>

// The model a subagent has when the router leaves it alone: the session's, by its id. A model above Opus
// (Fable, say) ranks above every route, so `model` is undefined: the router never names it.
function defaultOf(parentModel: string): { model: Model | undefined; rank: number } {
  const model = MODELS.find(name => parentModel.includes(name))
  return model === undefined ? { model, rank: MODELS.length } : { model, rank: RANK[model] }
}

// What a subagent starts on: `model` undefined keeps the session's; `floor`, the effort never drops below its own.
type Start = { model: Model | undefined; picked: Picked; floor: boolean }

// After a pushback the next subagent climbs one step: never below the session's model, one effort level above
// what it would have had, and only from Haiku, whose effort is fixed, to the next model.
function escalatedRoute(alone: Model | undefined, fallback: ReturnType<typeof defaultOf>, steps: Steps | null, lines: StepsLines): Start {
  const base = alone !== undefined && RANK[alone] > fallback.rank ? alone : fallback.model
  const model = base === 'haiku' ? 'sonnet' : base
  return { model, picked: { effort: raised(pickEffort(model ?? 'opus', steps, lines).effort), from: 'escalated' }, floor: true }
}

// Why the router leaves a subagent on the session's model, or undefined to route it: unsure of the answer, or a
// smaller model than the session's without the certainty that costs most when wrong (or the goal is quality).
function keepReason(isSure: boolean, isSmaller: boolean, confidence: number, smallerLine: number, goal: Settings['routerGoal']) {
  if (!isSure) return 'unsure'
  if (!isSmaller) return undefined
  if (goal === 'quality') return 'quality'
  return confidence < smallerLine ? 'downgrade' : undefined
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

// The session model each routed subagent was routed from, by its agent id: what its spend is compared against.
const routedFrom = atom({ plugin: 'vsr', key: 'routedFrom' } as const, {})

// Starts a subagent on `start`. The model is named only off the session's tier: an alias of the same tier would swap
// a pinned session model (claude-opus-4-8, say) for the alias's own.
async function spawnOn(
  $: EngineInterface,
  e: AgentSpawnInput,
  next: Parameters<Hook<'agent.spawn'>>[2],
  start: Start,
  fallback: ReturnType<typeof defaultOf>,
) {
  const isOwnTier = start.model === undefined || RANK[start.model] === fallback.rank
  const result = await next(isOwnTier ? e : { ...e, model: start.model })
  const { agentId } = result
  if (agentId !== undefined) {
    routes.set(agentId, { picked: start.picked, floor: start.floor })
    await update($, routedFrom, now => ({ ...now, [agentId]: e.parentModel }))
  }
  return result
}

export async function routeSpawn(
  $: EngineInterface,
  e: AgentSpawnInput,
  next: Parameters<Hook<'agent.spawn'>>[2],
  settings: Settings,
) {
  // A resumed subagent keeps the model and effort it ran on: the engine refuses a new model for it.
  const isLeftAlone = e.fork || e.isTeammate || e.resume !== undefined || e.model !== undefined || !ROUTED_TYPES.has(e.subagentType)
  if (!settings.apps.router || isLeftAlone) return next(e)
  const { router } = settings.thresholds
  const lines = { low: router.stepsLow, high: router.stepsHigh, minConfidence: router.stepsMinConfidence }
  const tiers = { easyBelow: router.easyBelow, hardFrom: router.hardFrom }
  const fallback = defaultOf(e.parentModel)
  const isEscalated = (await readEscalation($)) !== null
  const without = await read($, noScore)
  const asked =
    without === ''
      ? await classify($, settings.runtime, e.prompt, { complexity: COMPLEXITY, steps: STEPS }, QUICK_ANSWER_MS, parseAnswers)
      : { ok: false as const, why: `${without} answers no score questions`, latencyMs: 0 }
  const anchor = { id: e.tool_use_id, toolUseId: e.tool_use_id, at: await stamp($) }
  if (!asked.ok) {
    const why = isEscalated ? `${asked.why} · escalated, so one step up anyway` : asked.why
    await record($, settings.log, { kind: 'unavailable', app: 'router', ...anchor, subject: e.description, why, ...failureOf(asked) })
    return isEscalated ? spawnOn($, e, next, escalatedRoute(undefined, fallback, null, lines), fallback) : next(e)
  }
  const { complexity: { score, confidence }, steps } = asked.value
  const facts = { task: e.description, level: score, confidence, tiers, latencyMs: asked.latencyMs }
  const alone = routeOf(score, tiers)
  const isSure = confidence >= router.minConfidence
  const smallerLine = settings.routerGoal === 'save' ? router.minConfidence : Math.max(router.minConfidence, router.downgradeConfidence)
  const reason = isEscalated ? undefined : keepReason(isSure, RANK[alone.model] < fallback.rank, confidence, smallerLine, settings.routerGoal)
  if (reason !== undefined) {
    const result = await next(e)
    if (result.deny !== undefined) return result
    const minConfidence = reason === 'downgrade' ? smallerLine : router.minConfidence
    await record($, settings.log, { kind: 'keep', ...anchor, id: result.agentId ?? e.tool_use_id, model: result.model, minConfidence, reason, ...facts })
    return result
  }
  const start = isEscalated
    ? escalatedRoute(isSure ? alone.model : undefined, fallback, steps, lines)
    : { model: alone.model, picked: pickEffort(alone.model, steps, lines), floor: alone.floor }
  const result = await spawnOn($, e, next, start, fallback)
  if (result.deny !== undefined) return result
  await record($, settings.log, {
    kind: 'route',
    ...anchor,
    id: result.agentId ?? e.tool_use_id,
    agentId: result.agentId,
    // The model it runs on, as the engine started it: a session setting can override the router's.
    model: result.model,
    picked: start.picked.effort,
    steps,
    effortFrom: start.picked.from,
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
  const sessionModel = (await read($, routedFrom))[e.agentId]
  if (sessionModel === undefined) return
  const actual = costOf(e.usage.model, e.usage)
  const baseline = costOf(sessionModel, e.usage)
  if (actual === undefined || baseline === undefined) return
  await update($, spend, now => ({ actual: now.actual + actual, baseline: now.baseline + baseline, turns: now.turns + 1, sessionModel }))
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
