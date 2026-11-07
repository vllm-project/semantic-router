import type { EngineInterface, TurnCompleteInput, TurnStartInput } from 'claude-code'

import { classify, numberAt, offlineOf, type Reply } from './lib/classify'
import type { Settings } from './lib/settings'
import { EMPTY_TURN, readTurn, record, stamp, updateTurn } from './lib/state'
import { firstChars } from './lib/text'

// Past this the turn ends with no check (fail open).
const CHECK_TIMEOUT_MS = 3000
const MIN_ANSWER_CHARS = 40
const ANSWER_CHARS = 8000
// Enough of the answer's end to find its last drawn block again.
const TAIL_CHARS = 400

const HALLUCINATION = {
  hallucination: {
    type: 'noul',
    instructions:
      'Does the answer contain any factual claim that is unsupported by or contradicts the supplied grounding context? Judge the answer against context, not outside knowledge.',
  },
}

const parseHallucination = ({ answers }: Reply) => numberAt(answers.hallucination, 'noul')

export async function startTurnReads($: EngineInterface, e: TurnStartInput) {
  await updateTurn($, () => ({ ...EMPTY_TURN, prompt: e.text }))
}

// Checks a main-loop answer against what its turn read.
export async function checkTurn($: EngineInterface, e: TurnCompleteInput, settings: Settings) {
  if (!settings.apps.check) return
  const turn = await readTurn($)
  await updateTurn($, () => EMPTY_TURN)
  const answer = e.answer.trim()
  if (e.reason !== 'answer' || turn.files + turn.commands === 0 || answer.length < MIN_ANSWER_CHARS) return

  const state = { context: turn.context, request: turn.prompt, answer: firstChars(answer, ANSWER_CHARS) }
  const asked = await classify($, settings.runtime, state, HALLUCINATION, CHECK_TIMEOUT_MS, parseHallucination)
  const line = settings.thresholds.check.unsupported
  const at = await stamp($)
  const answerTail = answer.slice(-TAIL_CHARS)
  if (!asked.ok) {
    await record($, settings.log, { kind: 'unavailable', app: 'check', id: e.turnId, at, answerTail, subject: turn.prompt || 'the answer', why: asked.why, ...offlineOf(asked) })
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
