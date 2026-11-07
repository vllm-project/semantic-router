import { atom, read, update } from 'claude-code'
import type { EngineInterface } from 'claude-code'

import type { Decision, Escalation, TurnReads } from '../../types'
import { logDecision, relogDecision } from './log'
import type { Settings } from './settings'

const KEPT_DECISIONS = 100

// Change the tag when a record's fields change: it declines what an earlier version stored.
const decisions = atom({ plugin: 'vsr', key: 'decisions' } as const, [], {
  shape: 'decisions-7',
})

const escalation = atom({ plugin: 'vsr', key: 'escalation' } as const, null)

const lastPrompt = atom({ plugin: 'vsr', key: 'lastPrompt' } as const, null)

export const EMPTY_TURN: TurnReads = { prompt: '', files: 0, commands: 0, context: '' }

const turnReads = atom({ plugin: 'vsr', key: 'turn' } as const, EMPTY_TURN)

export async function stamp($: EngineInterface) {
  return new Date(await $.clock.now()).toISOString()
}

// Read while drawing, it redraws the site when the decisions change.
export async function readDecisions($: EngineInterface): Promise<readonly Decision[]> {
  return read($, decisions)
}

// Every decision is also a line of the decision log.
export async function record($: EngineInterface, log: Settings['log'], decision: Decision) {
  await update($, decisions, list => [...list, decision].slice(-KEPT_DECISIONS))
  await logDecision($, log, decision)
}

// Changes the decisions `isIt` picks, and rewrites their log lines.
export async function updateDecision(
  $: EngineInterface,
  log: Settings['log'],
  isIt: (decision: Decision) => boolean,
  change: (decision: Decision) => Decision,
) {
  const changed = (await read($, decisions)).filter(isIt).map(change)
  if (changed.length === 0) return
  await update($, decisions, list => list.map(one => (isIt(one) ? change(one) : one)))
  for (const one of changed) await relogDecision($, log, one)
}

export async function readEscalation($: EngineInterface): Promise<Escalation | null> {
  return read($, escalation)
}

export async function updateEscalation($: EngineInterface, change: (now: Escalation | null) => Escalation | null) {
  await update($, escalation, change)
}

// Keeps `text` as the latest prompt and answers the one before it.
export async function swapLastPrompt($: EngineInterface, text: string): Promise<string | null> {
  const prior = await read($, lastPrompt)
  await update($, lastPrompt, () => text)
  return prior
}

export async function readTurn($: EngineInterface): Promise<TurnReads> {
  return read($, turnReads)
}

export async function updateTurn($: EngineInterface, change: (turn: TurnReads) => TurnReads) {
  await update($, turnReads, change)
}
