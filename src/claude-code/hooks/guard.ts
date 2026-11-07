import type { EngineInterface, Hook, ToolCallResult } from 'claude-code'

import type { Decision, Outcome, Risk } from '../types'
import { classify, isRecord, numberAt, type Reply } from './lib/classify'
import type { Settings } from './lib/settings'
import { record, stamp, updateDecision } from './lib/state'
import { clip, fixed, oneLine, RISK_WORDS } from './lib/text'

// Past this the command goes on as it would have without the plugin (fail open).
const GUARD_TIMEOUT_MS = 3000
// Measured on vela3-lite on CPU: the runtime answers one part at a time, about 0.23 s each, so 12 parts fit in
// the 3 s; a command of more is held rather than judged on its first parts alone.
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

const RISKS: readonly Risk[] = ['safe', 'destructive', 'exfiltration', 'remote_code']

function parseGuard({ answers }: Reply) {
  const choice = isRecord(answers.kind) ? answers.kind.choice : undefined
  const risk = RISKS.find(one => one === choice)
  const riskProbability = risk && isRecord(answers.kind) ? numberAt(answers.kind.probabilities, risk) : undefined
  const review = numberAt(answers.review, 'noul')
  return risk === undefined || riskProbability === undefined || review === undefined
    ? undefined
    : { risk, riskProbability, review }
}

type Judged = { part: string } & NonNullable<ReturnType<typeof parseGuard>>

// `<<WORD`, `<<-WORD`, `<<'WORD'`, `<<"WORD"` or `<<\WORD`: a heredoc whose body runs to a line of WORD.
const HEREDOC = /^<<(-?)[ \t]*\\?(['"]?)([A-Za-z0-9_.-]+)\2/

// The index just past the line that ends each heredoc, its body read from `at`; the end of the command when one never ends.
function pastHeredocs(command: string, at: number, heredocs: readonly { word: string; isTabbed: boolean }[]) {
  for (const { word, isTabbed } of heredocs) {
    while (at < command.length) {
      const end = command.indexOf('\n', at)
      const line = command.slice(at, end < 0 ? command.length : end)
      at = end < 0 ? command.length : end + 1
      if ((isTabbed ? line.replace(/^\t+/, '') : line) === word) break
    }
  }
  return at
}

// The span of every separator that starts a command of its own: `&&`, `||`, `;`, a lone `&`, and a newline that
// no pipe continues. None inside quotes, `$(...)`, `(...)`, backticks, a comment or a heredoc's body.
function separatorsOf(command: string): [from: number, to: number][] {
  const cuts: [number, number][] = []
  const heredocs: { word: string; isTabbed: boolean }[] = []
  let quote: string | undefined
  let depth = 0
  let isTicked = false
  for (let i = 0; i < command.length; i++) {
    const c = command[i]
    const isTop = depth === 0 && !isTicked
    if (quote === "'") {
      if (c === "'") quote = undefined
      continue
    }
    if (c === '\\') {
      i++
      continue
    }
    if (quote === '"') {
      if (c === '"') quote = undefined
      continue
    }
    if (c === "'" || c === '"') quote = c
    else if (c === '`') isTicked = !isTicked
    else if (c === '(') depth++
    else if (c === ')') depth = Math.max(0, depth - 1)
    else if (c === '#' && (i === 0 || /[\s;&|()]/.test(command[i - 1]))) {
      const end = command.indexOf('\n', i)
      i = (end < 0 ? command.length : end) - 1
    } else if (command.startsWith('<<<', i)) i += 2
    else if (command.startsWith('<<', i)) {
      const marker = isTop ? HEREDOC.exec(command.slice(i)) : null
      if (marker !== null) heredocs.push({ word: marker[3], isTabbed: marker[1] === '-' })
      i += (marker?.[0].length ?? 2) - 1
    } else if (c === '\n' && heredocs.length > 0) {
      const past = pastHeredocs(command, i + 1, heredocs.splice(0))
      if (isTop && command[past - 1] === '\n') cuts.push([past - 1, past])
      i = past - 1
    } else if (isTop) {
      const width = separatorWidth(command, i)
      if (width > 0) cuts.push([i, i + width])
      i += Math.max(0, width - 1)
    }
  }
  return cuts
}

// How wide the separator at `i` is, or 0: `&` beside a redirection (`2>&1`, `&>`) or after a pipe (`|&`) is not one,
// nor is a newline after a pipe that runs on to the next line.
function separatorWidth(command: string, i: number) {
  if (command.startsWith('&&', i) || command.startsWith('||', i)) return 2
  const c = command[i]
  const before = command[i - 1] ?? ''
  if (c === ';') return 1
  if (c === '&') return /[<>|]/.test(before) || command[i + 1] === '>' ? 0 : 1
  if (c !== '\n') return 0
  const line = command.slice(0, i).trimEnd()
  return line.endsWith('|') && !line.endsWith('||') ? 0 : 1
}

// The commands a shell line runs, in order: a pipe stays whole, and a comment on a line of its own is left out.
export function partsOf(command: string): string[] {
  const parts: string[] = []
  let from = 0
  for (const [cutFrom, cutTo] of [...separatorsOf(command), [command.length, command.length] as const]) {
    parts.push(command.slice(from, cutFrom).trim())
    from = cutTo
  }
  return parts.filter(part => part !== '' && !part.startsWith('#'))
}

const TOP_DIRECTORIES = ['bin', 'boot', 'dev', 'etc', 'home', 'lib', 'lib64', 'opt', 'root', 'sbin', 'srv', 'usr', 'var']
// `/`, the home directory, or a top-level system directory, with or without a trailing `/` or `/*`.
const TOP = new RegExp(`^["']?(/|~|\\$HOME|\\$\\{HOME\\}|/(${TOP_DIRECTORIES.join('|')}))/?\\*?["']?$`)

// The words of a command, `sudo`, `env` and leading VAR=value assignments left off.
function wordsOf(part: string) {
  const words = part.trim().split(/\s+/)
  while (words.length > 0 && (words[0] === 'sudo' || words[0] === 'env' || /^[A-Za-z_][A-Za-z0-9_]*=/.test(words[0]))) words.shift()
  return words
}

const hasFlag = (flags: readonly string[], letter: string, long: string) =>
  flags.some(flag => flag === long || (/^-[^-]/.test(flag) && flag.includes(letter)))

// What makes a command catastrophic whatever the classifier scores it (a live `rm -rf ~` scored 0.78, under the
// line), or undefined: such commands are always held, also while the classifier is offline.
export function alwaysHeld(part: string): string | undefined {
  if (/:\s*\(\s*\)\s*\{[^}]*:\s*\|\s*:\s*&[^}]*\}\s*;\s*:/.test(part)) return 'starts a fork bomb'
  if (/\b(curl|wget)\b[^|]*\|\s*(sudo\s+)?(sh|bash|zsh|dash|ksh)\b/.test(part)) return 'runs a script straight from the internet'
  const [command = '', ...rest] = wordsOf(part)
  const name = command.split('/').pop() ?? ''
  const flags = rest.filter(word => word.startsWith('-'))
  const targets = rest.filter(word => !word.startsWith('-'))
  if (name === 'rm' && (hasFlag(flags, 'r', '--recursive') || hasFlag(flags, 'R', '--recursive'))) {
    if (targets.some(target => TOP.test(target))) return 'deletes a whole top-level directory'
  }
  if (name === 'mkfs' || name.startsWith('mkfs.')) return 'formats a disk'
  if (name === 'dd' && rest.some(word => word.startsWith('of=/dev/'))) return 'writes straight to a device'
  if (['chmod', 'chown', 'chgrp'].includes(name) && hasFlag(flags, 'R', '--recursive') && targets.includes('/')) {
    return 'changes every file on the machine'
  }
  return undefined
}

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
  const unchecked = Math.max(0, parts.length - MAX_PARTS)

  const started = await $.clock.now()
  const asked = await Promise.all(
    parts.slice(0, MAX_PARTS).map(async part => ({
      part,
      result: await classify($, settings.runtime, `Tool: Bash\nCommand: ${part}`, GUARD_QUESTIONS, GUARD_TIMEOUT_MS, parseGuard),
    })),
  )
  const latencyMs = (await $.clock.now()) - started
  const at = await stamp($)
  const id = e.tool_use_id ?? at

  const judged: Judged[] = []
  const failures: string[] = []
  for (const { part, result } of asked) {
    if (result.ok) judged.push({ part, ...result.value })
    else failures.push(result.why)
  }
  const scored = judged.reduce<Judged | undefined>(
    (top, one) => (top === undefined || one.review > top.review ? one : top),
    undefined,
  )
  const ruled = parts.flatMap(part => {
    const rule = alwaysHeld(part)
    return rule === undefined ? [] : [{ part, rule }]
  })[0]
  // A rule holds its part on its own; the classifier's score, when it gave one, is kept for the record (else 1).
  const riskiest: Judged | undefined = ruled
    ? (judged.find(one => one.part === ruled.part) ?? {
        part: ruled.part,
        risk: ruled.rule === 'runs a script straight from the internet' ? 'remote_code' : 'destructive',
        riskProbability: 1,
        review: 1,
      })
    : scored
  if (riskiest === undefined || (!ruled && riskiest.review < line && failures.length > 0 && unchecked === 0)) {
    const isOffline = asked.every(({ result }) => !result.ok && result.offline)
    await record($, settings.log, {
      kind: 'unavailable',
      app: 'guard',
      id,
      toolUseId: e.tool_use_id,
      at,
      subject: command,
      why: failures[0] ?? 'no part answered',
      ...(isOffline ? { offline: true as const } : {}),
    })
    return next(e)
  }
  const isRisky = riskiest.review >= line
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
  const reason = ruled
    ? `vSR guard: "${clip(oneLine(ruled.part), REASON_PART_WIDTH)}" ${ruled.rule}: always held, whatever the score. Confirm before it runs.`
    : isRisky
    ? `vSR guard: "${clip(oneLine(riskiest.part), REASON_PART_WIDTH)}" ${RISK_WORDS[riskiest.risk]} (risk score ${fixed(riskiest.review)} ≥ ${fixed(line)}). Confirm before it runs.`
    : `vSR guard: ${parts.length} commands in one call, more than the ${MAX_PARTS} it checks one at a time. Confirm before it runs.`
  return { decision: 'ask', reason }
}
