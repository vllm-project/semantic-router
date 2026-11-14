import { atom, read, update } from 'claude-code'
import type { EngineInterface, Hook, PromptSubmitInput, TurnCompleteInput, TurnStartInput } from 'claude-code'

import type { Escalation, Signal } from '../types'
import { classify, isRecord, numberAt, failureOf, QUICK_ANSWER_MS, type Reply } from './lib/classify'
import type { Settings } from './lib/settings'
import { fixed } from './lib/text'
import { record } from './lib/state'
import { stamp } from './lib/env'

// The pushback the next subagent climbs for, until the turn it started ends.
const escalation = atom({ plugin: 'vsr', key: 'escalation' } as const, null)
// The prompt before this one, for the repeat question.
const lastPrompt = atom({ plugin: 'vsr', key: 'lastPrompt' } as const, null)

// Read by the router at a spawn: whether the user just pushed back.
export async function readEscalation($: EngineInterface): Promise<Escalation | null> {
  return read($, escalation)
}

// Prompts a person wrote: a notification, a peer's message or a trigger is no feedback.
export const PERSON_ORIGINS = new Set(['composer', 'bridge', 'sdk', 'unclassified'])

const USER_FEEDBACK = {
  type: 'choice',
  instructions: 'What feedback does this user turn give about the previous answer?',
  require_full_input: true,
  choices: [
    { key: 'SAT', description: 'the user is satisfied with the previous answer' },
    { key: 'NEED_CLARIFICATION', description: 'the user asks for clarification or a more detailed explanation' },
    { key: 'WRONG_ANSWER', description: 'the user says the previous answer was wrong or did not work' },
    { key: 'WANT_DIFFERENT', description: 'the user wants a different answer, format, style or approach' },
    { key: 'NO_FEEDBACK', description: 'the user gives no feedback on the previous answer' },
  ],
}

const REASK = {
  type: 'noul',
  instructions:
    'Does the current user request seek the same information or action as the prior user request, including paraphrases and retries that add feedback? A different topic, objective or requested operation is not a repeat.',
}

const PUSHBACKS = [
  ['WRONG_ANSWER', 'wrong'],
  ['WANT_DIFFERENT', 'different'],
] as const

// Every `user_feedback` probability; none without the two a pushback is read from.
function parseFeedback({ answers }: Reply): Record<string, number> | undefined {
  const probabilities = isRecord(answers.user_feedback) ? answers.user_feedback.probabilities : undefined
  if (!isRecord(probabilities) || PUSHBACKS.some(([key]) => numberAt(probabilities, key) === undefined)) return undefined
  return Object.fromEntries(Object.entries(probabilities).filter((one): one is [string, number] => typeof one[1] === 'number'))
}

const parseReask = ({ answers }: Reply) => numberAt(answers.reask, 'noul')

function signalsOf(feedback: Record<string, number> | null, reask: number | null, { thresholds }: Settings): Signal[] {
  const { feedback: line, reask: reaskLine } = thresholds.escalate
  const pushed = PUSHBACKS.flatMap(([key, kind]) => {
    const score = feedback?.[key] ?? 0
    return score >= line ? [{ kind, score, threshold: line }] : []
  })
  return reask !== null && reask >= reaskLine ? [...pushed, { kind: 'repeat', score: reask, threshold: reaskLine }] : pushed
}

const WHY: Record<Signal['kind'], string> = {
  wrong: 'they said it was wrong',
  different: 'they want a different answer',
  repeat: 'they asked for the same thing again',
}

export async function escalatePrompt(
  $: EngineInterface,
  e: PromptSubmitInput,
  next: Parameters<Hook<'prompt.submit'>>[2],
  settings: Settings,
) {
  if (!settings.apps.escalate || e.isMeta || e.command || e.text.trim() === '' || !PERSON_ORIGINS.has(e.origin.kind)) return next(e)
  const prior = await read($, lastPrompt)
  await update($, lastPrompt, () => e.text)
  const [feedback, repeat] = await Promise.all([
    classify($, settings.runtime, e.text, { user_feedback: USER_FEEDBACK }, QUICK_ANSWER_MS, parseFeedback),
    prior === null ? undefined : classify($, settings.runtime, { current: e.text, prior }, { reask: REASK }, QUICK_ANSWER_MS, parseReask),
  ])
  const asked = repeat === undefined ? [feedback] : [feedback, repeat]
  const probabilities = feedback.ok ? feedback.value : null
  const reask = repeat?.ok ? repeat.value : null
  const signals = signalsOf(probabilities, reask, settings)
  const at = await stamp($)
  if (signals.length === 0) {
    await update($, escalation, () => null)
    const failed = asked.find(one => !one.ok)
    if (failed !== undefined && !failed.ok) {
      await record($, settings.log, { kind: 'unavailable', app: 'escalate', id: at, at, prompt: e.text, subject: e.text, why: failed.why, ...failureOf(failed) })
    }
    return next(e)
  }
  await update($, escalation, () => ({ prompt: e.text, signals, turnId: e.turnId }))
  const latencyMs = Math.max(...asked.map(one => one.latencyMs))
  const turns = settings.apps.router && settings.routerTurns ? { turns: true as const } : {}
  await record($, settings.log, { kind: 'escalate', id: at, at, prompt: e.text, signals, feedback: probabilities, reask, latencyMs, ...turns })
  const [first] = signals
  const note = `vSR: the user pushed back on your previous answer (${WHY[first.kind]}, ${fixed(first.score)}). Re-check that answer before you build on it.`
  return next({ ...e, context: [...(e.context ?? []), note] })
}

// The first main turn to start after a pushback is the one that answers it.
export async function holdEscalation($: EngineInterface, e: TurnStartInput) {
  const held = await read($, escalation)
  if (held !== null && held.turnId === undefined) {
    await update($, escalation, now => (now === null ? null : { ...now, turnId: e.turnId }))
  }
}

export async function releaseEscalation($: EngineInterface, e: TurnCompleteInput) {
  const held = await read($, escalation)
  if (held !== null && held.turnId === e.turnId) await update($, escalation, () => null)
}
