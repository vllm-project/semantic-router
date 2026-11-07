import { atom, read, update } from 'claude-code'
import type { EngineInterface } from 'claude-code'

import type { Decision } from '../../types'
import { logDecision, relogDecision } from './log'
import type { Settings } from './settings'

const KEPT_DECISIONS = 100

// Change the tag when a record's fields change: it declines what an earlier version stored.
const decisions = atom({ plugin: 'vsr', key: 'decisions' } as const, [], {
  shape: 'decisions-7',
})


// Read while drawing, it redraws the site when the decisions change.
export async function readDecisions($: EngineInterface): Promise<readonly Decision[]> {
  return read($, decisions)
}

// Every decision is also a line of the decision log. A call the user cancelled is not one: nothing went unchecked.
export async function record($: EngineInterface, log: Settings['log'], decision: Decision) {
  if (decision.kind === 'unavailable' && decision.cancelled) return
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

