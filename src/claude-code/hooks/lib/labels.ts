import { atom, read, update } from 'claude-code'
import type { EngineInterface } from 'claude-code'

import type { Decision } from '../../types'
import { fileOf } from './log'
import type { Settings } from './settings'

// What the person can say of a decision in the pane: a binary app's call was right or wrong; a route's model was
// right, too weak or too strong.
export type Answer = 'right' | 'wrong' | 'too-low' | 'too-high'

export const ANSWERS: Readonly<Record<'binary' | 'router', readonly { answer: Answer; label: string }[]>> = {
  binary: [
    { answer: 'right', label: 'right call' },
    { answer: 'wrong', label: 'wrong call' },
  ],
  router: [
    { answer: 'right', label: 'right model' },
    { answer: 'too-low', label: 'needed stronger' },
    { answer: 'too-high', label: 'could be lighter' },
  ],
}

// The labels given in this session, by decision id, so the pane marks the chosen answer.
const given = atom({ plugin: 'vsr', key: 'labels' } as const, {})

export async function readLabels($: EngineInterface): Promise<Readonly<Record<string, Answer>>> {
  return read($, given)
}

// Whether sr acted on a decision: the side of its line the score fell on.
const acted = (d: Decision) =>
  d.kind === 'guard' ? d.verdict === 'ask' : d.kind === 'check' ? d.verdict === 'unverified' : true

// Which answers a decision takes, or undefined for one calibration does not learn from: a hold by a fixed rule or
// for too many parts, a redaction by the secret rules alone, an escalated route, a skipped call.
export function answersFor(d: Decision): 'binary' | 'router' | undefined {
  switch (d.kind) {
    case 'route':
      return d.escalated ? undefined : 'router'
    case 'keep':
      return 'router'
    case 'guard':
      return d.rule || d.unchecked ? undefined : 'binary'
    case 'redact':
      return Object.keys(d.counts).every(tag => tag === 'secret') ? undefined : 'binary'
    case 'shield':
    case 'escalate':
    case 'check':
      return 'binary'
    default:
      return undefined
  }
}

// The label calibration reads for an answer: a router's as given; a binary app's whether it should have acted.
export function labelOf(d: Decision, answer: Answer): string {
  if (answersFor(d) === 'router') return answer
  return acted(d) === (answer === 'right') ? 'yes' : 'no'
}

// Appends the label beside the decision log, where /vsr:calibrate reads it. Never throws.
export async function giveLabel($: EngineInterface, log: Settings['log'], d: Decision, answer: Answer) {
  await update($, given, labels => ({ ...labels, [d.id]: answer }))
  if (log.path === '') return
  try {
    const logFile = await fileOf($, log.path)
    if (logFile === undefined) return
    const file = logFile.replace(/[^/]*$/, 'labels.jsonl')
    let before = ''
    try {
      before = await $.fs.read(file)
    } catch {
      // No labels yet.
    }
    await $.fs.write(file, `${before}${JSON.stringify({ id: d.id, app: d.kind, label: labelOf(d, answer), at: new Date(await $.clock.now()).toISOString() })}\n`)
  } catch {
    // A label that cannot be written is still marked in the pane.
  }
}
