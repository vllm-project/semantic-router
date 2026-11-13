import { atom, read, update } from 'claude-code'
import type { EngineInterface, Hook, PromptSubmitInput, TurnCompleteInput, TurnStartInput, TurnStepInput, TurnUsage } from 'claude-code'

import type { Effort, EffortFrom, Model, Reread, Steps, TurnContext, TurnRoute } from '../types'
import { classify, failureOf, QUICK_ANSWER_MS, type Classified } from './lib/classify'
import { effortFor, pickEffort, raised } from './lib/effort'
import { stamp } from './lib/env'
import { costOf, PAYBACK_TURNS, rereadCost } from './lib/prices'
import type { Settings } from './lib/settings'
import { record, updateDecision } from './lib/state'
import { modelName } from './lib/text'
import { PERSON_ORIGINS, readEscalation } from './escalate'
import { addSpend, COMPLEXITY, defaultOf, MODELS, parseAnswers, RANK, readNoScore, routeOf, STEPS } from './router'

// Your own turns, routed like subagents (`routerTurns`): the router reads the prompt you send and picks the model and
// effort the turn runs on, on your own Claude login. Unlike a gateway that switches on every request, a turn keeps its
// model through its tool calls, and a move between turns is weighed against the prompt cache it gives up.

// A reply this short follows the turn before it ("yes, do it"): it never moves your turns to a smaller model.
export const SHORT_REPLY_WORDS = 12


const turnRoute = atom({ plugin: 'vsr', key: 'turnRoute' } as const, null)
const turnPaused = atom({ plugin: 'vsr', key: 'turnPaused' } as const, '')
const turnContext = atom({ plugin: 'vsr', key: 'turnContext' } as const, null)
const lastTurnUsage = atom({ plugin: 'vsr', key: 'lastTurnUsage' } as const, null)

type Asked = Classified<{ complexity: Steps; steps: Steps | null }>

// The prompt you just sent and the router's question on it, asked as it is submitted so the turn waits for it once. A
// turn that starts on another text (a continuation, a notification) is not a prompt to route.
let asked: { text: string; answer: Promise<Asked> } | undefined

// What each of your turns runs on, by turn id, from its start to its end: the route, the decision to update with the
// effort its first request ran at, and whether its spend counts as routed.
const bound = new Map<string, { route: TurnRoute | null; decisionId?: string; session: string }>()

const isRouting = (settings: Settings) => settings.apps.router && settings.routerTurns

// Read while drawing: where your turns run, and the model you picked with /model when that paused the router.
export async function readTurns($: EngineInterface) {
  return { route: await read($, turnRoute), paused: await read($, turnPaused) }
}

// The status line's entry while the router routes your turns; none once /vsr turns off reloaded the plugin.
function showStatus($: EngineInterface, settings: Settings, route: TurnRoute | null, paused: string) {
  if (!isRouting(settings)) $.ui.status(undefined)
  else if (paused !== '') $.ui.status(`vSR ⇢ paused · you picked ${modelName(paused)}`)
  else $.ui.status(route === null ? 'vSR ⇢ your turns on your model' : `vSR ⇢ your turns on ${modelName(route.model)} · ${route.picked}`)
}

export async function showTurns($: EngineInterface, settings: Settings) {
  const { route, paused } = await readTurns($)
  showStatus($, settings, route, paused)
}

// `/vsr turns on`: follow the router again from the model the session is on now.
export async function resumeTurns($: EngineInterface, settings: Settings) {
  await update($, turnPaused, () => '')
  await update($, turnRoute, () => null)
  showStatus($, settings, null, '')
}

export function askTurn($: EngineInterface, e: PromptSubmitInput, settings: Settings) {
  asked = undefined
  if (!isRouting(settings) || e.isMeta || e.command || e.text.trim() === '' || !PERSON_ORIGINS.has(e.origin.kind)) return
  const answer = (async (): Promise<Asked> => {
    const without = await readNoScore($)
    if (without !== '') return { ok: false, why: `${without} answers no score questions`, latencyMs: 0 }
    return classify($, settings.runtime, e.text, { complexity: COMPLEXITY, steps: STEPS }, QUICK_ANSWER_MS, parseAnswers)
  })()
  asked = { text: e.text, answer }
}

// What the router decides for a turn: move to another model, keep the one the turns run on and say why, or stay
// without a word (the answer names the tier they are on already).
export type TurnPlan =
  | { kind: 'move'; model: Model; picked: TurnRoute['picked']; effortFrom: TurnRoute['effortFrom']; floor: boolean; escalated: boolean }
  | { kind: 'keep'; reason: 'unsure' | 'downgrade' | 'quality' | 'follow-up' | 'cache'; would: Model; minConfidence: number; reread?: Reread }
  | { kind: 'stay' }

export type TurnFacts = {
  level: number
  confidence: number
  steps: Steps | null
  words: number
  // The session's model, as /model shows it, and the tier the turns run on now (undefined: the session's own).
  session: string
  now: Model | undefined
  isEscalated: boolean
  context: TurnContext | null
  last: TurnUsage | null
}

const nameOf = (model: Model | undefined, session: string) => model ?? session

// The model at a rank, Opus for any rank above it.
const tierAt = (rank: number): Model => MODELS[Math.max(0, Math.min(RANK.opus, rank))] ?? 'opus'

export function planTurn(facts: TurnFacts, settings: Settings): TurnPlan {
  const { router } = settings.thresholds
  const tiers = { easyBelow: router.easyBelow, hardFrom: router.hardFrom }
  const lines = { low: router.stepsLow, high: router.stepsHigh, minConfidence: router.stepsMinConfidence }
  const session = defaultOf(facts.session)
  const nowRank = facts.now === undefined ? session.rank : RANK[facts.now]
  const routed = routeOf(facts.level, tiers)
  // Haiku runs your turns only with the save goal, or when the session is on it already: a coding agent's own loop
  // needs more than a subagent's errand.
  const floorRank = settings.routerGoal === 'save' ? 0 : Math.min(RANK.sonnet, session.rank)
  const target = tierAt(Math.max(RANK[routed.model], floorRank))
  if (facts.isEscalated) {
    // A pushback climbs one step from where the last turn ran, at one effort level more; above Opus there is no step.
    const base = Math.max(nowRank, RANK[target])
    if (base > RANK.opus) return { kind: 'stay' }
    const up = tierAt(base + 1)
    return { kind: 'move', model: up, picked: raised(pickEffort(up, facts.steps, lines).effort), effortFrom: 'escalated', floor: true, escalated: true }
  }
  const rank = RANK[target]
  if (rank === nowRank) return { kind: 'stay' }
  if (facts.confidence < router.minConfidence) return { kind: 'keep', reason: 'unsure', would: target, minConfidence: router.minConfidence }
  const picked = pickEffort(target, facts.steps, lines)
  const move = { kind: 'move' as const, model: target, picked: picked.effort, effortFrom: picked.from, floor: routed.floor, escalated: false }
  if (rank > nowRank) return move
  if (rank < session.rank) {
    if (settings.routerGoal === 'quality') return { kind: 'keep', reason: 'quality', would: target, minConfidence: router.minConfidence }
    const smallerLine = settings.routerGoal === 'save' ? router.minConfidence : Math.max(router.minConfidence, router.downgradeConfidence)
    if (facts.confidence < smallerLine) return { kind: 'keep', reason: 'downgrade', would: target, minConfidence: smallerLine }
  }
  if (facts.words <= SHORT_REPLY_WORDS) return { kind: 'keep', reason: 'follow-up', would: target, minConfidence: router.minConfidence }
  // A move to a smaller model re-reads the conversation without the cache: it waits while that costs more than
  // PAYBACK_TURNS turns like the last one would save.
  if (facts.context !== null && facts.last !== null) {
    const from = nameOf(facts.now, facts.session)
    const cost = rereadCost(from, target, facts.context.tokens)
    const [was, would] = [costOf(from, facts.last), costOf(target, facts.last)]
    if (cost !== undefined && was !== undefined && would !== undefined && cost > (was - would) * PAYBACK_TURNS) {
      return { kind: 'keep', reason: 'cache', would: target, minConfidence: router.minConfidence, reread: { tokens: facts.context.tokens, cost, saving: (was - would) * PAYBACK_TURNS } }
    }
  }
  return move
}

export async function startTurn($: EngineInterface, e: TurnStartInput, settings: Settings) {
  if (e.agentId !== undefined) return
  const mine = asked !== undefined && asked.text === e.text ? asked : undefined
  asked = undefined
  if (!isRouting(settings)) return
  const session = await $.session.model()
  const paused = await read($, turnPaused)
  let current = await read($, turnRoute)
  if (paused === '' && current !== null && current.anchor !== session) {
    // You picked another model with /model: the turns follow it until you hand them back.
    await update($, turnPaused, () => session)
    await update($, turnRoute, () => null)
    $.ui.toast(`vSR router: you picked ${modelName(session)} with /model, so your turns stay on it · /vsr turns on routes them again`)
    showStatus($, settings, null, session)
    return
  }
  if (paused !== '') return
  if (mine === undefined) {
    // A turn you did not type runs where the last one did.
    bound.set(e.turnId, { route: current, session })
    return
  }
  const answer = await mine.answer
  const at = await stamp($)
  const anchor = { id: at, at, scope: 'turn' as const, prompt: e.text, task: e.text }
  if (!answer.ok) {
    await record($, settings.log, { kind: 'unavailable', app: 'router', id: at, at, prompt: e.text, subject: e.text, why: answer.why, ...failureOf(answer) })
    bound.set(e.turnId, { route: current, session })
    return
  }
  const { complexity, steps } = answer.value
  const { router } = settings.thresholds
  const tiers = { easyBelow: router.easyBelow, hardFrom: router.hardFrom }
  const escalation = await readEscalation($)
  const plan = planTurn(
    {
      level: complexity.score,
      confidence: complexity.confidence,
      steps,
      words: e.text.trim().split(/\s+/).length,
      session,
      now: current?.model,
      isEscalated: escalation !== null && escalation.prompt === e.text,
      context: await read($, turnContext),
      last: await read($, lastTurnUsage),
    },
    settings,
  )
  const facts = { level: complexity.score, confidence: complexity.confidence, tiers, latencyMs: answer.latencyMs }
  if (plan.kind === 'keep') {
    const model = nameOf(current?.model, session)
    await record($, settings.log, { kind: 'keep', ...anchor, model, would: plan.would, minConfidence: plan.minConfidence, reason: plan.reason, reread: plan.reread, ...facts })
  }
  if (plan.kind === 'move') {
    const from = nameOf(current?.model, session)
    // Back on the session's tier the turn runs as the session would, its own model and effort; a pushback still
    // raises the effort there.
    const isSession = RANK[plan.model] === defaultOf(session).rank && !plan.escalated
    current = isSession ? null : { model: plan.model, picked: plan.picked, effortFrom: plan.effortFrom, floor: plan.floor, anchor: session }
    await update($, turnRoute, () => current)
    await record($, settings.log, {
      kind: 'route',
      ...anchor,
      from,
      model: current === null ? session : current.model,
      picked: plan.picked,
      steps,
      effortFrom: current === null ? 'floor' : plan.effortFrom,
      sessionModel: session,
      ...(plan.escalated ? { escalated: { alone: 'default' as const } } : {}),
      ...facts,
    })
    bound.set(e.turnId, { route: current, decisionId: at, session })
    showStatus($, settings, current, '')
    return
  }
  bound.set(e.turnId, { route: current, session })
  showStatus($, settings, current, '')
}

export async function* stepTurn($: EngineInterface, e: TurnStepInput, next: Parameters<Hook<'turn.step'>>[2], settings: Settings) {
  const turn = bound.get(e.turnId)
  const route = turn?.route ?? null
  let request = e
  let ran: { effort: Effort; from?: EffortFrom } | undefined = typeof e.effort === 'string' ? { effort: e.effort } : undefined
  if (route !== null) {
    // The session's own tier is not named: its alias would swap a pinned model (claude-opus-4-8, say) for the latest.
    const isOwnTier = turn !== undefined && RANK[route.model] === defaultOf(turn.session).rank
    request = isOwnTier ? e : { ...e, model: route.model }
    if (typeof e.effort === 'string') {
      const { effort, from } = effortFor({ effort: route.picked, from: route.effortFrom }, route.floor, e.effort)
      request = { ...request, effort }
      ran = { effort, from }
    }
  }
  const decisionId = turn?.decisionId
  if (e.index === 0 && decisionId !== undefined && ran !== undefined) {
    const { effort, from } = ran
    await updateDecision($, settings.log, d => d.id === decisionId, d => (d.kind === 'route' ? { ...d, effort, ...(from === undefined ? {} : { effortFrom: from }) } : d))
  }
  const result = yield* next(request)
  if (result.usage !== null) {
    const { input_tokens, cache_read_input_tokens, cache_creation_input_tokens, model } = result.usage
    await update($, turnContext, () => ({ tokens: input_tokens + cache_read_input_tokens + cache_creation_input_tokens, model }))
  }
  return result
}

export async function endTurn($: EngineInterface, e: TurnCompleteInput) {
  if (e.agentId !== undefined) return
  const turn = bound.get(e.turnId)
  bound.delete(e.turnId)
  if (e.usage === undefined) return
  await update($, lastTurnUsage, () => e.usage ?? null)
  if (turn?.route !== null && turn?.route !== undefined) await addSpend($, e.usage, turn.session)
}
