import { atom, read, update } from 'claude-code'
import type { EngineInterface } from 'claude-code'

import type { Decision } from '../../types'
import { appOf } from './apps'
import { fileOf, labelsFileOf, stamp } from './env'
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

// Whether vSR acted on a decision: the side of its line the score fell on.
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

// The labels file, or undefined where no log is kept to label: the pane offers answers only where they are kept.
export async function labelsFile($: EngineInterface, log: Settings['log']) {
  const logFile = log.path === '' ? undefined : await fileOf($, log.path)
  return logFile === undefined ? undefined : labelsFileOf(logFile)
}

// One label line, the shape /vsr:calibrate reads and writes too.
async function lineOf($: EngineInterface, d: Decision, answer: Answer) {
  return `${JSON.stringify({ id: d.id, app: appOf(d), label: labelOf(d, answer), at: await stamp($) })}\n`
}

// Writes run one after another, so two quick presses both land.
let writes: Promise<void> = Promise.resolve()

// Appends one line: a file that exists but cannot be read is left as it is rather than replaced by this line alone.
async function append($: EngineInterface, file: string, line: string) {
  const before = (await $.fs.exists(file)) ? await $.fs.read(file) : ''
  await $.fs.write(file, `${before}${line}`)
}

// Marks the answer in the pane and appends the label beside the decision log, where /vsr:calibrate reads it. Never throws.
export async function giveLabel($: EngineInterface, log: Settings['log'], d: Decision, answer: Answer) {
  await update($, given, labels => ({ ...labels, [d.id]: answer }))
  const file = await labelsFile($, log)
  if (file === undefined) return
  const line = await lineOf($, d, answer)
  writes = writes.then(() => append($, file, line)).catch(() => undefined)
  await writes
}
