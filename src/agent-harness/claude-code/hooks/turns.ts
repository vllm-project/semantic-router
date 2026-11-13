import { atom, read, update } from 'claude-code'
import type {
  ClassicEventOf,
  EngineInterface,
  Hook,
  PromptSubmitInput,
  TurnCompleteInput,
  TurnStartInput,
  TurnStepInput,
  TurnUsage,
} from 'claude-code'

import type { Effort, EffortFrom, Model, Reread, Steps, TurnContext, TurnRoute } from '../types'
import { classify, failureOf, QUICK_ANSWER_MS, type Classified } from './lib/classify'
import { EFFORTS, effortFor, pickEffort, raised } from './lib/effort'
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

// A reply this short follows the turn before it ("yes, do it"): the router reads it with that prompt, and it never moves
// your turns to a smaller model.
export const SHORT_REPLY_WORDS = 12

// The prompt cache's default lifetime: a conversation idle longer than this has nothing warm to lose by a move.
export const CACHE_WARM_MS = 5 * 60_000

const turnRoute = atom({ plugin: 'vsr', key: 'turnRoute' } as const, null)
const turnBound = atom({ plugin: 'vsr', key: 'turnBound' } as const, null)
const turnPaused = atom({ plugin: 'vsr', key: 'turnPaused' } as const, '')
const turnContext = atom({ plugin: 'vsr', key: 'turnContext' } as const, null)
const lastTurnUsage = atom({ plugin: 'vsr', key: 'lastTurnUsage' } as const, null)
const turnPrompt = atom({ plugin: 'vsr', key: 'turnPrompt' } as const, '')

type Asked = Classified<{ complexity: Steps; steps: Steps | null }>

// Your prompts and the router's question on each, asked as each is submitted so its turn waits for it once, by text: a
// prompt queued behind a running turn keeps its own answer. A turn that starts on another text (a continuation, a
// notification) is not a prompt to route.
const asked = new Map<string, Promise<Asked>>()

const isRouting = (settings: Settings) => settings.apps.router && settings.routerTurns
const wordsIn = (text: string) => text.trim().split(/\s+/).length

// Read while drawing: where your turns run, and the model you picked yourself when that paused the router.
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

// A model you pick yourself (/model, the picker, an SDK's set_model) stands until /vsr turns on; a fallback the engine
// makes or a resumed session's model does not.
const PICKED = new Set(['command', 'picker', 'sdk'])

export async function modelPicked($: EngineInterface, e: ClassicEventOf['classic.PostModelSwitch'], settings: Settings) {
  if (!isRouting(settings) || !PICKED.has(e.source)) return
  await update($, turnPaused, () => e.to_model)
  await update($, turnRoute, () => null)
  $.ui.toast(`vSR router: you picked ${modelName(e.to_model)}, so your turns stay on it · /vsr turns on routes them again`)
  showStatus($, settings, null, e.to_model)
}

export function askTurn($: EngineInterface, e: PromptSubmitInput, settings: Settings) {
  if (!isRouting(settings) || e.isMeta || e.command || e.text.trim() === '' || !PERSON_ORIGINS.has(e.origin.kind)) return
  const answer = (async (): Promise<Asked> => {
    const prior = await read($, turnPrompt)
    await update($, turnPrompt, () => e.text)
    const without = await readNoScore($)
    if (without !== '') return { ok: false, why: `${without} answers no score questions`, latencyMs: 0 }
    // "yes, do it" is as hard as what it agrees to.
    const state = wordsIn(e.text) <= SHORT_REPLY_WORDS && prior !== '' ? `${prior}\n\n${e.text}` : e.text
    return classify($, settings.runtime, state, { complexity: COMPLEXITY, steps: STEPS }, QUICK_ANSWER_MS, parseAnswers)
  })()
  asked.set(e.text, answer)
}

// The effort a turn starts at when it keeps its tier: none when the steps answer is unsure, so it keeps the last one.
type Pick = { picked: Effort; effortFrom: TurnRoute['effortFrom']; floor: boolean }

// What the router decides for a turn: move to another model, keep the one the turns run on and say why, or stay without
// a word (the answer names the tier they are on already); a keep or a stay can still set the turn's effort.
export type TurnPlan =
  | ({ kind: 'move'; model: Model; escalated: boolean } & Pick)
  | { kind: 'keep'; reason: 'unsure' | 'downgrade' | 'quality' | 'follow-up' | 'cache'; would: Model; minConfidence: number; reread?: Reread; effort?: Pick }
  | { kind: 'stay'; effort?: Pick }

export type TurnFacts = {
  level: number
  confidence: number
  steps: Steps | null
  words: number
  // The session's model, as /model shows it, and the tier the turns run on now (undefined: the session's own).
  session: string
  now: Model | undefined
  isEscalated: boolean
  // The conversation the cache holds warm, and the last turn's usage; null when nothing is warm.
  context: TurnContext | null
  last: TurnUsage | null
  // The effort the last request ran at.
  lastEffort: Effort | undefined
}

const nameOf = (model: Model | undefined, session: string) => model ?? session

// The model at a rank, Opus for any rank above it.
const tierAt = (rank: number): Model => MODELS[Math.max(0, Math.min(RANK.opus, rank))] ?? 'opus'

const higher = (one: Effort, other: Effort | undefined) => (other !== undefined && EFFORTS.indexOf(other) > EFFORTS.indexOf(one) ? other : one)

// A session on a 1M-token window (`opus[1m]`) keeps it on the model its turns move to: the plain alias has a smaller
// window than the conversation may already fill. Haiku has no such window, so it never runs those turns.
const isLongWindow = (session: string) => /\[1m\]$/i.test(session)
const aliasOf = (model: Model, session: string) => (isLongWindow(session) ? `${model}[1m]` : model)

export function planTurn(facts: TurnFacts, settings: Settings): TurnPlan {
  const { router } = settings.thresholds
  const tiers = { easyBelow: router.easyBelow, hardFrom: router.hardFrom }
  const lines = { low: router.stepsLow, high: router.stepsHigh, minConfidence: router.stepsMinConfidence }
  const session = defaultOf(facts.session)
  const nowRank = facts.now === undefined ? session.rank : RANK[facts.now]
  const nowTier = facts.now ?? session.model
  const routed = routeOf(facts.level, tiers)
  // Haiku runs your turns only with the save goal, or when the session is on it already: a coding agent's own loop
  // needs more than a subagent's errand.
  const floorRank = settings.routerGoal === 'save' && !isLongWindow(facts.session) ? 0 : Math.min(RANK.sonnet, session.rank)
  const target = tierAt(Math.max(RANK[routed.model], floorRank))
  if (facts.isEscalated) {
    // A pushback climbs one step from where the last turn ran, and one effort level above what it ran at; above Opus
    // there is no step.
    const base = Math.max(nowRank, RANK[target])
    if (base > RANK.opus) return { kind: 'stay' }
    const up = tierAt(base + 1)
    return { kind: 'move', model: up, picked: raised(higher(pickEffort(up, facts.steps, lines).effort, facts.lastEffort)), effortFrom: 'escalated', floor: true, escalated: true }
  }
  // A turn that keeps its tier still starts at the effort its prompt needs, unless the answer is unsure or the prompt
  // is a short reply, which goes on as the turn before it did.
  const steady = nowTier === undefined ? undefined : pickEffort(nowTier, facts.steps, lines)
  const effort =
    steady === undefined || steady.from === 'unsure' || facts.words <= SHORT_REPLY_WORDS || nowTier === undefined
      ? undefined
      : { picked: steady.effort, effortFrom: steady.from, floor: nowTier === 'opus' }
  const keep = (reason: Extract<TurnPlan, { kind: 'keep' }>['reason'], minConfidence: number, reread?: Reread): TurnPlan => ({
    kind: 'keep',
    reason,
    would: target,
    minConfidence,
    ...(reread === undefined ? {} : { reread }),
    ...(effort === undefined ? {} : { effort }),
  })
  const rank = RANK[target]
  if (rank === nowRank) return effort === undefined ? { kind: 'stay' } : { kind: 'stay', effort }
  if (facts.confidence < router.minConfidence) return keep('unsure', router.minConfidence)
  const picked = pickEffort(target, facts.steps, lines)
  const move: TurnPlan = { kind: 'move', model: target, picked: picked.effort, effortFrom: picked.from, floor: routed.floor, escalated: false }
  if (rank > nowRank) return move
  if (rank < session.rank) {
    if (settings.routerGoal === 'quality') return keep('quality', router.minConfidence)
    const smallerLine = settings.routerGoal === 'save' ? router.minConfidence : Math.max(router.minConfidence, router.downgradeConfidence)
    if (facts.confidence < smallerLine) return keep('downgrade', smallerLine)
  }
  if (facts.words <= SHORT_REPLY_WORDS) return keep('follow-up', router.minConfidence)
  // A move to a smaller model writes the whole conversation to its cache once: it waits while that costs more than
  // PAYBACK_TURNS turns like the last one would save there.
  if (facts.context !== null && facts.last !== null) {
    const cost = rereadCost(target, facts.context.tokens)
    const [was, would] = [costOf(nameOf(facts.now, facts.session), facts.last), costOf(target, facts.last)]
    if (cost !== undefined && was !== undefined && would !== undefined && cost > (was - would) * PAYBACK_TURNS) {
      return keep('cache', router.minConfidence, { tokens: facts.context.tokens, cost, saving: (was - would) * PAYBACK_TURNS })
    }
  }
  return move
}

async function bind($: EngineInterface, turnId: string, route: TurnRoute | null, session: string, decisionId?: string) {
  await update($, turnBound, () => ({ turnId, route, session, ...(decisionId === undefined ? {} : { decisionId }) }))
}

export async function startTurn($: EngineInterface, e: TurnStartInput, settings: Settings) {
  if (e.agentId !== undefined) return
  const mine = asked.get(e.text)
  asked.delete(e.text)
  await update($, turnBound, () => null)
  if (!isRouting(settings) || (await read($, turnPaused)) !== '') return
  const session = await $.session.model()
  let current = await read($, turnRoute)
  // A turn you did not type runs where the last one did.
  if (mine === undefined) return bind($, e.turnId, current, session)
  const answer = await mine
  const at = await stamp($)
  const anchor = { id: at, at, scope: 'turn' as const, prompt: e.text, task: e.text }
  if (!answer.ok) {
    await record($, settings.log, { kind: 'unavailable', app: 'router', id: at, at, prompt: e.text, subject: e.text, why: answer.why, ...failureOf(answer) })
    return bind($, e.turnId, current, session)
  }
  const { complexity, steps } = answer.value
  const { router } = settings.thresholds
  const tiers = { easyBelow: router.easyBelow, hardFrom: router.hardFrom }
  const escalation = await readEscalation($)
  const context = await read($, turnContext)
  const isSameConversation = context !== null && context.session === (await $.session.id())
  // A conversation /clear started over, or one idle past the cache's lifetime, has nothing warm to lose.
  const isWarm = isSameConversation && (await $.clock.now()) - context.at < CACHE_WARM_MS
  const plan = planTurn(
    {
      level: complexity.score,
      confidence: complexity.confidence,
      steps,
      words: wordsIn(e.text),
      session,
      now: current?.model,
      isEscalated: escalation !== null && escalation.prompt === e.text,
      context: isWarm ? context : null,
      last: isWarm ? await read($, lastTurnUsage) : null,
      lastEffort: isSameConversation ? context.effort : undefined,
    },
    settings,
  )
  const facts = { level: complexity.score, confidence: complexity.confidence, tiers, latencyMs: answer.latencyMs }
  const tier = current?.model ?? defaultOf(session).model
  if (plan.kind !== 'move') {
    if (plan.kind === 'keep') {
      const { would, minConfidence, reason, reread } = plan
      await record($, settings.log, { kind: 'keep', ...anchor, model: nameOf(current?.model, session), would, minConfidence, reason, ...(reread ? { reread } : {}), ...facts })
    }
    if (plan.effort !== undefined && tier !== undefined) {
      current = { model: tier, ...plan.effort }
      await update($, turnRoute, () => current)
    }
    await bind($, e.turnId, current, session)
    showStatus($, settings, current, '')
    return
  }
  const from = nameOf(current?.model, session)
  current = { model: plan.model, picked: plan.picked, effortFrom: plan.effortFrom, floor: plan.floor }
  await update($, turnRoute, () => current)
  await record($, settings.log, {
    kind: 'route',
    ...anchor,
    from,
    // Back on the session's tier the turn runs on the session's own model, a pinned one included.
    model: RANK[plan.model] === defaultOf(session).rank ? session : plan.model,
    picked: plan.picked,
    steps,
    effortFrom: plan.effortFrom,
    sessionModel: session,
    ...(plan.escalated ? { escalated: { alone: 'default' as const } } : {}),
    ...facts,
  })
  await bind($, e.turnId, current, session, at)
  showStatus($, settings, current, '')
}

export async function* stepTurn($: EngineInterface, e: TurnStepInput, next: Parameters<Hook<'turn.step'>>[2], settings: Settings) {
  const bound = await read($, turnBound)
  const turn = bound?.turnId === e.turnId ? bound : null
  const route = turn?.route ?? null
  let request = e
  let ran: { effort: Effort; from?: EffortFrom } | undefined = typeof e.effort === 'string' ? { effort: e.effort } : undefined
  if (turn !== null && route !== null) {
    // The session's own tier is not named: its alias would swap a pinned model (claude-opus-4-8, say) for the latest.
    const isOwnTier = RANK[route.model] === defaultOf(turn.session).rank
    request = isOwnTier ? e : { ...e, model: aliasOf(route.model, turn.session) }
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
    // The next request re-sends this one's prompt and its answer.
    const { input_tokens, cache_read_input_tokens, cache_creation_input_tokens, output_tokens, model } = result.usage
    const tokens = input_tokens + cache_read_input_tokens + cache_creation_input_tokens + output_tokens
    const [session, now] = [await $.session.id(), await $.clock.now()]
    const effort = typeof request.effort === 'string' ? request.effort : undefined
    await update($, turnContext, () => ({ tokens, model, session, at: now, ...(effort === undefined ? {} : { effort }) }))
  }
  return result
}

export async function endTurn($: EngineInterface, e: TurnCompleteInput) {
  if (e.agentId !== undefined) return
  const bound = await read($, turnBound)
  if (bound?.turnId === e.turnId) await update($, turnBound, () => null)
  if (e.usage === undefined) return
  await update($, lastTurnUsage, () => e.usage ?? null)
  // A turn counts as routed when it ran on another tier than the session's.
  const route = bound?.turnId === e.turnId ? bound.route : null
  if (route !== null && RANK[route.model] !== defaultOf(bound?.session ?? '').rank) await addSpend($, e.usage, bound?.session ?? '')
}
