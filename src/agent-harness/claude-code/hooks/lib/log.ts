import { atom, read, update } from 'claude-code'
import type { EngineInterface } from 'claude-code'

import type { App, Calibration, Decision, EffortFrom, Model, Outcome, Risk, Steps, Tag, Tiers } from '../../types'
import { isRecord } from './classify'
import { fileOf } from './env'
import { appOf } from './apps'
import { answersFor } from './labels'
import { hide, secretsIn } from './redact'
import type { Settings } from './settings'
import { clip, oneLine } from './text'

export const KEPT_LINES = 5000
const SUBJECT_CHARS = 200
// `$.store` keys: the newest lines of every session, and when the thresholds were last calibrated.
const LINES = 'log'
const CALIBRATED_AT = 'calibratedAt'

const calibration = atom({ plugin: 'vsr', key: 'calibration' } as const, { at: null, since: 0, isReminded: false })

export type Verdict = 'routed' | 'kept' | 'escalated' | 'hidden' | 'shielded' | 'held' | 'allowed' | 'grounded' | 'unverified' | 'off'

export type Detail =
  | {
      level: number
      confidence: number
      model: string
      effort: string | null
      // The steps answer, and where the effort came from; null on a kept subagent, which has no route.
      steps: Steps | null
      effortFrom: EffortFrom | null
      escalated: { alone: Model | 'default' } | null
    }
  | { feedback: Record<string, number> | null; reask: number | null }
  | { counts: Partial<Record<Tag, number>>; source: string }
  | { source: string }
  | { risk: Risk; riskProbability: number; review: number; parts: number; outcome: Outcome | null }
  | { files: number; commands: number }
  | { why: string }

// One line of the decision log, the calibration skill's only input. `score` is the number held against `threshold`; both null for `off`.
// `learnable`: whether calibration learns from it (answersFor), so the log, the pane and the skill share one rule.
export type LogLine = {
  id: string
  at: string
  session: string
  app: App
  verdict: Verdict
  score: number | null
  threshold: number | null
  subject: string
  detail: Detail
  learnable: boolean
}

type Judged = Pick<LogLine, 'verdict' | 'score' | 'threshold' | 'detail'> & { subject: string }

// The tier line a routed level sits nearest: the one a calibration would move to route it otherwise.
const nearestLine = (level: number, { easyBelow, hardFrom }: Tiers) =>
  Math.abs(level - easyBelow) <= Math.abs(level - hardFrom) ? easyBelow : hardFrom

const lines: { [K in Decision['kind']]: (decision: Extract<Decision, { kind: K }>) => Judged } = {
  route: d => ({
    verdict: 'routed',
    score: d.level,
    threshold: nearestLine(d.level, d.tiers),
    subject: d.task,
    detail: {
      level: d.level,
      confidence: d.confidence,
      model: d.model,
      effort: d.effort ?? null,
      steps: d.steps,
      effortFrom: d.effortFrom,
      escalated: d.escalated ?? null,
    },
  }),
  keep: d => ({
    verdict: 'kept',
    score: d.confidence,
    threshold: d.minConfidence,
    subject: d.task,
    detail: { level: d.level, confidence: d.confidence, model: d.model, effort: null, steps: null, effortFrom: null, escalated: null, reason: d.reason ?? 'unsure' },
  }),
  escalate: d => ({
    verdict: 'escalated',
    score: d.signals[0].score,
    threshold: d.signals[0].threshold,
    subject: d.prompt,
    detail: { feedback: d.feedback, reask: d.reask },
  }),
  redact: d => ({
    verdict: 'hidden',
    score: d.score,
    threshold: d.threshold,
    subject: d.source,
    detail: { counts: d.counts, source: d.source },
  }),
  shield: d => ({
    verdict: 'shielded',
    score: d.score,
    threshold: d.threshold,
    subject: d.source,
    detail: { source: d.source },
  }),
  guard: d => ({
    verdict: d.verdict === 'ask' ? 'held' : 'allowed',
    score: d.review,
    threshold: d.threshold,
    subject: d.part,
    detail: {
      risk: d.risk,
      riskProbability: d.riskProbability,
      review: d.review,
      parts: d.parts,
      outcome: d.outcome ?? null,
      ...(d.unchecked ? { unchecked: d.unchecked } : {}),
      ...(d.rule ? { rule: d.rule } : {}),
    },
  }),
  check: d => ({
    verdict: d.verdict,
    score: d.score,
    threshold: d.threshold,
    subject: d.prompt || 'the answer',
    detail: { files: d.files, commands: d.commands },
  }),
  unavailable: d => ({ verdict: 'off', score: null, threshold: null, subject: d.subject, detail: { why: d.why } }),
}

// A subject never carries what redact hides by rule.
const subjectOf = (text: string) => clip(oneLine(hide(text, secretsIn(text)).text), SUBJECT_CHARS)

export function lineOf(decision: Decision, session: string): LogLine {
  const judged = (lines[decision.kind] as (decision: Decision) => Judged)(decision)
  return { id: decision.id, at: decision.at, session, app: appOf(decision), ...judged, subject: subjectOf(judged.subject), learnable: answersFor(decision) !== undefined }
}

const isSame = (a: LogLine, b: LogLine) => a.id === b.id && a.app === b.app && a.session === b.session

const isLine = (value: unknown): value is LogLine =>
  isRecord(value) && typeof value.id === 'string' && typeof value.at === 'string' && typeof value.app === 'string'

const linesIn = (value: unknown): LogLine[] => (Array.isArray(value) ? value.filter(isLine) : [])

const calibratedAtIn = (value: unknown) => (typeof value === 'string' ? value : null)

async function sessionOf($: EngineInterface) {
  try {
    return await $.session.id()
  } catch {
    return ''
  }
}

// Whether a file is this log's to write: absent, empty, or starting with a log line. Another file at the path is
// left alone, not replaced by the log; asked once per file per session.
const owned = new Map<string, Promise<boolean>>()

async function isOwned($: EngineInterface, file: string) {
  try {
    if (!(await $.fs.exists(file))) return true
    const first = (await $.fs.read(file)).split('\n', 1)[0]
    return first === '' || isLine(JSON.parse(first))
  } catch {
    return false
  }
}

// Keeps the newest lines the store takes: past its size limit, the older half goes until they fit.
async function store($: EngineInterface, all: LogLine[]) {
  for (let kept = all.slice(-KEPT_LINES); ; kept = kept.slice(Math.ceil(kept.length / 2))) {
    try {
      await $.store.set(LINES, kept)
      return kept
    } catch (error) {
      if (kept.length <= 1) throw error
    }
  }
}

const sinceCalibrated = (all: readonly LogLine[], at: string | null) => all.filter(line => at === null || line.at > at).length

// Draws the count since the last calibration, and toasts once a session when it reaches `calibrateEvery`.
async function remind($: EngineInterface, log: Settings['log'], all: readonly LogLine[]) {
  const at = calibratedAtIn(await $.store.get(CALIBRATED_AT))
  const since = sinceCalibrated(all, at)
  const { isReminded } = await read($, calibration)
  const isDue = log.calibrateEvery > 0 && since >= log.calibrateEvery && !isReminded
  await update($, calibration, () => ({ at, since, isReminded: isReminded || isDue }))
  if (isDue) $.ui.toast(`vSR: ${since} decisions since the last calibration · run /vsr:calibrate`)
}

type Change = (all: LogLine[]) => LogLine[]

// Changes wait here while a write runs; the next write takes them all, so a burst of decisions is one write.
const pending: Change[] = []
let writes: Promise<void> = Promise.resolve()
// A write still running after this long is left behind, so one that never finishes cannot hold back every later
// line; the next write rewrites the whole log from the store.
const WRITE_MS = 10_000

async function flush($: EngineInterface, log: Settings['log']) {
  if (pending.length === 0) return
  const batch = pending.splice(0)
  const all = await store($, batch.reduce((now, change) => change(now), linesIn(await $.store.get(LINES))))
  await remind($, log, all)
  const file = await fileOf($, log.path)
  if (file === undefined) return
  if (!owned.has(file)) owned.set(file, isOwned($, file))
  if (await owned.get(file)) await $.fs.write(file, all.map(line => `${JSON.stringify(line)}\n`).join(''))
}

function changeLog($: EngineInterface, log: Settings['log'], change: Change) {
  pending.push(change)
  writes = writes.then(() => Promise.race([flush($, log), $.clock.sleep(WRITE_MS)])).catch(() => undefined)
}

// Writes a new decision's line, the hook that made it not waiting on the write. Never throws.
export async function logDecision($: EngineInterface, log: Settings['log'], decision: Decision) {
  if (log.path === '') return
  const line = lineOf(decision, await sessionOf($))
  changeLog($, log, all => [...all, line])
}

// Rewrites the newest line of a decision that changed (a guard's outcome, a route's effort). Never throws.
export async function relogDecision($: EngineInterface, log: Settings['log'], decision: Decision) {
  if (log.path === '') return
  const line = lineOf(decision, await sessionOf($))
  changeLog($, log, all => {
    const at = all.findLastIndex(one => isSame(one, line))
    return at < 0 ? all : all.with(at, line)
  })
}

// Read while drawing: the pane's calibration line.
export async function readCalibration($: EngineInterface): Promise<Calibration> {
  return read($, calibration)
}

// At session start: the count since the last calibration, and its reminder when due.
export async function loadCalibration($: EngineInterface, log: Settings['log']) {
  if (log.path === '') return
  await remind($, log, linesIn(await $.store.get(LINES)))
}

// `/vsr calibrated`: the calibration skill applied new thresholds now.
export async function markCalibrated($: EngineInterface, at: string) {
  await $.store.set(CALIBRATED_AT, at)
  const since = sinceCalibrated(linesIn(await $.store.get(LINES)), at)
  await update($, calibration, now => ({ ...now, at, since }))
}
