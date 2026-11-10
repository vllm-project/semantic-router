import { atom, read, update } from 'claude-code'
import type { EngineInterface, ToolCallInput, ToolCallResult, TurnCompleteInput, TurnStartInput } from 'claude-code'

import type { TurnReads } from '../types'

import { ANSWER_MS, classify, numberAt, failureOf, type Reply } from './lib/classify'
import type { Settings } from './lib/settings'
import { record } from './lib/state'
import { stamp } from './lib/env'
import { firstChars } from './lib/text'
import { outputOf, sourceOf } from './output'

const MIN_ANSWER_CHARS = 40
const ANSWER_CHARS = 8000
// Enough of the answer's end to find its last drawn block again.
const TAIL_CHARS = 400
// What check keeps of a turn's reads, in characters.
const KEPT_CONTEXT_CHARS = 8000
// The tools whose output an answer is checked against.
const CHECKED = new Set(['Read', 'Bash'])

const EMPTY_TURN: TurnReads = { prompt: '', files: 0, commands: 0, context: '' }
const turnReads = atom({ plugin: 'vsr', key: 'turn' } as const, EMPTY_TURN)

const HALLUCINATION = {
  hallucination: {
    type: 'noul',
    instructions:
      'Does the answer contain any factual claim that is unsupported by or contradicts the supplied grounding context? Judge the answer against context, not outside knowledge.',
  },
}

const parseHallucination = ({ answers }: Reply) => numberAt(answers.hallucination, 'noul')

export async function startTurnReads($: EngineInterface, e: TurnStartInput) {
  await update($, turnReads, () => ({ ...EMPTY_TURN, prompt: e.text }))
}

// Keeps what a main-loop Read or Bash gave Claude (after redact), for the check at the turn's end.
export async function noteRead($: EngineInterface, e: ToolCallInput, ran: ToolCallResult, settings: Settings) {
  const tool = String(e.tool)
  if (!settings.apps.check || !CHECKED.has(tool) || e.agentId !== undefined) return
  const fields = outputOf(tool, ran)?.fields.filter(field => field.trim() !== '') ?? []
  if (fields.length === 0) return
  const read = `${await sourceOf($, e)}:\n${fields.join('\n')}`
  await update($, turnReads, turn => ({
    ...turn,
    files: turn.files + (tool === 'Read' ? 1 : 0),
    commands: turn.commands + (tool === 'Bash' ? 1 : 0),
    context: firstChars(turn.context === '' ? read : `${turn.context}\n\n${read}`, KEPT_CONTEXT_CHARS),
  }))
}

// Checks a main-loop answer against what its turn read.
export async function checkTurn($: EngineInterface, e: TurnCompleteInput, settings: Settings) {
  if (!settings.apps.check) return
  const turn = await read($, turnReads)
  await update($, turnReads, () => EMPTY_TURN)
  const answer = e.answer.trim()
  if (e.reason !== 'answer' || turn.files + turn.commands === 0 || answer.length < MIN_ANSWER_CHARS) return

  const state = { context: turn.context, request: turn.prompt, answer: firstChars(answer, ANSWER_CHARS) }
  const asked = await classify($, settings.runtime, state, HALLUCINATION, ANSWER_MS, parseHallucination)
  const line = settings.thresholds.check.unsupported
  const at = await stamp($)
  const answerTail = answer.slice(-TAIL_CHARS)
  if (!asked.ok) {
    await record($, settings.log, { kind: 'unavailable', app: 'check', id: e.turnId, at, answerTail, subject: turn.prompt || 'the answer', why: asked.why, ...failureOf(asked) })
    return
  }
  await record($, settings.log, {
    kind: 'check',
    id: e.turnId,
    at,
    verdict: asked.value < line ? 'grounded' : 'unverified',
    score: asked.value,
    threshold: line,
    files: turn.files,
    commands: turn.commands,
    prompt: turn.prompt,
    answerTail,
    latencyMs: asked.latencyMs,
  })
}
