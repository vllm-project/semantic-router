import type { EngineInterface, Hook, ToolCallResult } from 'claude-code'

import type { Decision, Outcome, Risk } from '../types'
import { type Classified, classify, failureOf, isRecord, numberAt, type Reply } from './lib/classify'
import { budgetOf, type Settings } from './lib/settings'
import { stamp } from './lib/env'
import { alwaysHeld, firstOf, partsOf } from './lib/shell'
import { record, updateDecision } from './lib/state'
import { clip, fixed, oneLine, RISK_WORDS } from './lib/text'

// Measured on vela3-lite on CPU: the runtime answers one part at a time, about 0.23 s each, so 12 parts fit in
// the default 3 s wait; a command of more is held rather than judged on its first parts alone.
const MAX_PARTS = 12
const REASON_PART_WIDTH = 80

const GUARD_QUESTIONS = {
  kind: {
    type: 'choice',
    instructions: 'What is the main risk of running this shell command?',
    require_full_input: true,
    choices: [
      { key: 'safe', description: 'a routine command that reads, builds, tests or makes ordinary reversible changes' },
      {
        key: 'destructive',
        description: 'deletes, overwrites or corrupts important data or system state in a way that is hard to undo',
      },
      { key: 'exfiltration', description: 'sends local files, secrets or credentials to an outside server' },
      { key: 'remote_code', description: 'downloads code from the internet and runs it' },
    ],
  },
  review: {
    type: 'noul',
    instructions:
      'Should a person review this shell command before it runs, because it could cause serious harm that is hard to undo?',
  },
}

const RISKS = Object.keys(RISK_WORDS) as Risk[]

function parseGuard({ answers }: Reply) {
  const choice = isRecord(answers.kind) ? answers.kind.choice : undefined
  const risk = RISKS.find(one => one === choice)
  const riskProbability = risk && isRecord(answers.kind) ? numberAt(answers.kind.probabilities, risk) : undefined
  const review = numberAt(answers.review, 'noul')
  return risk === undefined || riskProbability === undefined || review === undefined
    ? undefined
    : { risk, riskProbability, review }
}

type Guarded = NonNullable<ReturnType<typeof parseGuard>>
type Judged = { part: string } & Guarded

// A refusal reaches the hook as `deny` or, at the dialog and from rules, as an errored result. A shell command that ran and failed reads `Exit code N`.
export const isRefusal = (ran: ToolCallResult) =>
  ran.deny !== undefined || (ran.isError === true && !/^Exit code \d/.test(ran.text ?? ''))

const isHeldCall = (d: Decision, id: string) => d.kind === 'guard' && d.verdict === 'ask' && d.id === id

// A held call's outcome, once its call settled: run, or refused by the person or a rule.
export async function settleHeld($: EngineInterface, id: string | undefined, outcome: Outcome, settings: Settings) {
  if (id === undefined) return
  await updateDecision($, settings.log, d => isHeldCall(d, id), d => ({ ...d, outcome }))
}

export async function guardCheck(
  $: EngineInterface,
  e: Parameters<Hook<'tool.check'>>[1],
  next: Parameters<Hook<'tool.check'>>[2],
  settings: Settings,
) {
  const command = isRecord(e.input) && typeof e.input.command === 'string' ? e.input.command : undefined
  if (!settings.apps.guard || e.isReadOnly || command === undefined || command.trim() === '') return next(e)
  const line = settings.thresholds.guard.review

  const parts = partsOf(command)
  if (parts.length === 0) return next(e)
  const overLimit = Math.max(0, parts.length - MAX_PARTS)

  // The runtime answers one request at a time, so the parts go one after another within one budget; once one
  // reaches the line the command is held whatever the rest score, and the rest are not asked.
  const budget = budgetOf(settings.waits.guard)
  const started = await $.clock.now()
  const asked: { part: string; result: Classified<Guarded> }[] = []
  for (const part of parts.slice(0, MAX_PARTS)) {
    const left = budget - ((await $.clock.now()) - started)
    const result: Classified<Guarded> =
      left > 0
        ? await classify($, settings.runtime, `Tool: Bash\nCommand: ${part}`, GUARD_QUESTIONS, left, parseGuard)
        : { ok: false, why: `no time left of the ${budget} ms`, latencyMs: 0 }
    asked.push({ part, result })
    if (result.ok && result.value.review >= line) break
  }
  const latencyMs = (await $.clock.now()) - started
  const at = await stamp($)
  const id = e.tool_use_id ?? at

  const judged: Judged[] = []
  const failures: { part: string; why: string; offline?: true; cancelled?: true }[] = []
  for (const { part, result } of asked) {
    if (result.ok) judged.push({ part, ...result.value })
    else failures.push({ part, why: result.why, ...failureOf(result) })
  }
  const scored = judged.reduce<Judged | undefined>(
    (top, one) => (top === undefined || one.review > top.review ? one : top),
    undefined,
  )
  const ruled = firstOf(parts, part => {
    const held = alwaysHeld(part)
    return held && { part, ...held }
  })
  const isRisky = scored !== undefined && scored.review >= line
  // A part the decision model could not check (offline, slow or an error answer): with a wait of seconds vSR steps
  // aside, as redact and shield do, and only the fixed rules and a score already over the line hold; a guard that
  // waits until done holds instead. Cancelled with the call (the user pressed Esc): nothing will run, so nothing is
  // held or recorded.
  if (failures.length > 0 && failures.every(failure => failure.cancelled) && !ruled && !isRisky) return next(e)
  const isSteppingAside = failures.length > 0 && settings.waits.guard !== 'done'
  if (isSteppingAside && !ruled && !isRisky && overLimit === 0) {
    const failure = failures.find(one => !one.cancelled) ?? failures[0]
    await record($, settings.log, {
      kind: 'unavailable',
      app: 'guard',
      id,
      toolUseId: e.tool_use_id,
      at,
      subject: command,
      why: failure.why,
      ...(failure.offline ? { offline: true as const } : {}),
    })
    return next(e)
  }
  const notChecked = isSteppingAside ? 0 : failures.length
  const unchecked = overLimit + notChecked
  // A rule holds its part on its own; the classifier's score, when it gave one, is kept for the record (else 1).
  // With no part answered, the record names the first part not checked, at no score.
  const riskiest: Judged = ruled
    ? (judged.find(one => one.part === ruled.part) ?? { part: ruled.part, risk: ruled.risk, riskProbability: 1, review: 1 })
    : (scored ?? { part: failures[0].part, risk: 'safe', riskProbability: 0, review: 0 })
  const isHeld = ruled !== undefined || isRisky || unchecked > 0
  await record($, settings.log, {
    kind: 'guard',
    id,
    at,
    command,
    part: riskiest.part,
    parts: parts.length,
    verdict: isHeld ? 'ask' : 'pass',
    risk: riskiest.risk,
    riskProbability: riskiest.riskProbability,
    review: riskiest.review,
    threshold: line,
    latencyMs,
    ...(unchecked > 0 ? { unchecked } : {}),
    ...(ruled ? { rule: ruled.rule } : {}),
  })
  const decided = await next(e)
  // The guard only tightens: a refusal from elsewhere stands.
  if (!isHeld || decided.decision === 'deny') return decided
  const steps = parts.length === 1 ? 'step' : 'steps'
  const reason = ruled
    ? `vSR guard: "${clip(oneLine(ruled.part), REASON_PART_WIDTH)}" ${ruled.rule}: always held, whatever the score. Confirm before it runs.`
    : isRisky
    ? `vSR guard: "${clip(oneLine(riskiest.part), REASON_PART_WIDTH)}" ${RISK_WORDS[riskiest.risk]} (risk score ${fixed(riskiest.review)} ≥ ${fixed(line)}). Confirm before it runs.`
    : notChecked > 0
    ? `vSR guard: could not check ${unchecked} of ${parts.length} ${steps}: ${failures[0].why}. Confirm before it runs.`
    : `vSR guard: ${parts.length} commands in one call, more than the ${MAX_PARTS} it checks one at a time. Confirm before it runs.`
  return { decision: 'ask', reason }
}
