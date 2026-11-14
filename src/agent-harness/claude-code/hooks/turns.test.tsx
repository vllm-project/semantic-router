import { expect, test } from 'claude-code/testing'
import type { On, TurnUsage } from 'claude-code'

import { engine, logLines, paneText, type Reply, rowTree, type Test, userRow } from './kit'
import { parseSettings } from './lib/settings'
import { planTurn, type TurnFacts } from './turns'

const HARD = 'Find why the cache test fails only under load, prove the race with a trace, and fix it without a lock.'
const EASY = 'Print the names of the files in the docs folder of this repository, one per line, nothing else please.'
const YES = 'yes, do it'
const TURNS = { options: { routerTurns: true, display: 'verbose' } } as const

// The model the session is on, as /model would show it; `session.model` reads it at every turn's start.
function session(on: On, model: { now: string }) {
  on('session.model', () => ({ value: model.now }))
}

// The requests each turn made, with the model and effort they named, and the usage each one answers with.
function steps(on: On, usage?: Partial<TurnUsage>) {
  const made: { turnId: string; model: string; effort?: unknown }[] = []
  on('turn.step', async function* (_, e) {
    made.push({ turnId: e.turnId, model: e.model, effort: e.effort })
    const full = usage === undefined ? null : { input_tokens: 0, output_tokens: 0, cache_read_input_tokens: 0, cache_creation_input_tokens: 0, model: e.model, ...usage }
    return { turnId: e.turnId, index: e.index, answer: '', toolUses: [], stopReason: 'end_turn', usage: full }
  })
  return made
}

const submit = ($: Test, text: string) => $.prompt.submit({ text, origin: { kind: 'composer' } } as never)

// One of your turns: the prompt, its start, `requests` steps on the session's model at medium effort, and its end.
async function turn($: Test, turnId: string, text: string, requests = 2, usage?: TurnUsage) {
  if (text !== '') await submit($, text)
  await $.turn.start({ turnId, text })
  for (let index = 0; index < requests; index++) {
    for await (const _ of $.turn.step({ turnId, index, model: 'claude-sonnet-5-5', effort: 'medium', messageCount: 1 + index })) void _
  }
  await $.turn.complete({ turnId, answer: 'done', usage } as never)
}

test('your turns are not routed unless routerTurns is on', { options: { display: 'verbose' } }, async ($, on) => {
  const reply: Reply = { complexity: { score: 1.8, confidence: 0.9 }, bodies: [] }
  engine(on, reply)
  session(on, { now: 'claude-sonnet-5-5' })
  const made = steps(on)

  await turn($, 't1', HARD)

  expect(made.map(one => one.model)).toEqual(['claude-sonnet-5-5', 'claude-sonnet-5-5'])
  expect(reply.bodies?.some(body => 'complexity' in body.questions)).toBe(false)
})

test('a hard prompt moves the turn to Opus, every request of it, and its card stands under the prompt', TURNS, async ($, on) => {
  const reply: Reply = { complexity: { score: 1.8, confidence: 0.9 }, steps: { score: 1.4, confidence: 0.8 }, statuses: [] }
  engine(on, reply)
  session(on, { now: 'claude-sonnet-5-5' })
  const made = steps(on)

  await turn($, 't1', HARD)

  expect(made).toEqual([
    { turnId: 't1', model: 'opus', effort: 'xhigh' },
    { turnId: 't1', model: 'opus', effort: 'xhigh' },
  ])
  const { texts } = await rowTree($, userRow(HARD))
  expect(texts).toContain('⇢  ROUTED  your turn → Opus · xhigh effort')
  expect(texts).toContain('reasoning steps 1.40 → xhigh · from Sonnet 5.5')
  expect(reply.statuses?.at(-1)).toBe('vSR ⇢ your turns on Opus · xhigh')
})

test('an easy prompt goes to Sonnet, not Haiku, unless the goal is save', TURNS, async ($, on) => {
  const reply: Reply = { complexity: { score: 0.2, confidence: 0.95 }, steps: { score: 0.2, confidence: 0.8 }, files: {} }
  engine(on, reply)
  session(on, { now: 'claude-opus-5-5' })
  const made = steps(on)

  await turn($, 't1', EASY)

  expect(made[0]).toEqual({ turnId: 't1', model: 'sonnet', effort: 'low' })
  // The log tells a turn's route from a subagent's, and says where it moved from.
  expect(logLines(reply).at(-1)?.detail).toMatchObject({ level: 0.2, confidence: 0.95, model: 'sonnet', scope: 'turn', from: 'claude-opus-5-5' })
})

test('with the save goal an easy prompt goes to Haiku', { options: { routerTurns: true, routerGoal: 'save' } }, async ($, on) => {
  engine(on, { complexity: { score: 0.2, confidence: 0.95 } })
  session(on, { now: 'claude-opus-5-5' })
  const made = steps(on)

  await turn($, 't1', EASY)

  expect(made[0]?.model).toBe('haiku')
})

test('a short reply keeps the model the turn before it ran on, and is read with the prompt before it', TURNS, async ($, on) => {
  const reply: Reply = { complexity: { score: 1.8, confidence: 0.9 }, files: {}, bodies: [] }
  engine(on, reply)
  session(on, { now: 'claude-sonnet-5-5' })
  const made = steps(on)

  await turn($, 't1', HARD, 1)
  reply.complexity = { score: 0.1, confidence: 0.97 }
  await turn($, 't2', YES, 1)

  expect(made.map(one => one.model)).toEqual(['opus', 'opus'])
  const { texts } = await rowTree($, userRow(YES))
  expect(texts).toContain('≈  KEPT  your turn stays on Opus')
  // No threshold decided it, so calibration does not learn from it.
  expect(logLines(reply).at(-1)?.learnable).toBe(false)
  // a second short reply still reads the task, not the reply before it
  await turn($, 't3', 'continue', 1)
  const asked = reply.bodies?.filter(body => 'complexity' in body.questions).map(body => body.state)
  expect(asked).toEqual([HARD, `${HARD}\n\n${YES}`, `${HARD}\n\ncontinue`])
  expect(texts).toContain("short reply: keeps the last turn's model, not Sonnet")
})

test('a move to a smaller model waits while re-reading the conversation costs more than it saves', TURNS, async ($, on) => {
  const reply: Reply = { complexity: { score: 1.8, confidence: 0.9 } }
  engine(on, reply)
  session(on, { now: 'claude-opus-5-5' })
  const made = steps(on, { input_tokens: 500, cache_read_input_tokens: 100_000, output_tokens: 2000 })
  const last = { input_tokens: 500, output_tokens: 2000, cache_read_input_tokens: 100_000, cache_creation_input_tokens: 0, model: 'claude-opus-5-5' }

  await turn($, 't1', HARD, 1, last)
  reply.complexity = { score: 0.9, confidence: 0.95 }
  await turn($, 't2', EASY, 1)

  expect(made.map(one => one.model)).toEqual(['claude-sonnet-5-5', 'claude-sonnet-5-5'])
  const { texts } = await rowTree($, userRow(EASY))
  expect(texts).toContain('≈  KEPT  your turn stays on Opus 5.5')
  expect(texts).toContain('Sonnet would re-read 103K tokens uncached: $0.24, 3 turns save $0.06')
})

// What Claude Code reports after the session's model changed, and who changed it.
const switched = (to: string, source: string) =>
  ({ hook_event_name: 'PostModelSwitch', from_model: 'claude-sonnet-5-5', to_model: to, requested_model: null, source, context_tokens: 0, prompt_cache_warm: true, cache_ttl: '5m', estimated_cache_write_usd: 0, pricing: 'catalog', session_id: 's', transcript_path: '/t', cwd: '/repo' }) as never

test('a conversation idle past the cache lifetime moves without weighing a re-read', TURNS, async ($, on) => {
  const reply: Reply = { complexity: { score: 1.8, confidence: 0.9 } }
  const clock = engine(on, reply)
  session(on, { now: 'claude-opus-5-5' })
  const made = steps(on, { input_tokens: 500, cache_read_input_tokens: 100_000, output_tokens: 2000 })
  const last = { input_tokens: 500, output_tokens: 2000, cache_read_input_tokens: 100_000, cache_creation_input_tokens: 0, model: 'claude-opus-5-5' }

  await turn($, 't1', HARD, 1, last)
  await clock.advance(6 * 60_000)
  reply.complexity = { score: 0.9, confidence: 0.95 }
  await turn($, 't2', EASY, 1)

  expect(made.at(-1)?.model).toBe('sonnet')
})

test('a model you pick yourself pauses the router until /vsr turns on', TURNS, async ($, on) => {
  const reply: Reply = { complexity: { score: 1.8, confidence: 0.9 }, toasts: [], statuses: [] }
  engine(on, reply)
  on('classic.PostModelSwitch', () => ({}))
  const model = { now: 'claude-sonnet-5-5' }
  session(on, model)
  const made = steps(on)

  await turn($, 't1', HARD, 1)
  model.now = 'claude-haiku-5-5'
  await $.classic.PostModelSwitch(switched('claude-haiku-5-5', 'picker'))
  await turn($, 't2', HARD, 1)

  expect(made.map(one => one.model)).toEqual(['opus', 'claude-sonnet-5-5'])
  expect(reply.toasts).toContain('vSR router: you picked Haiku 5.5, so your turns stay on it · /vsr turns on routes them again')
  expect(reply.statuses?.at(-1)).toBe('vSR ⇢ paused · you picked Haiku 5.5')
  expect(await paneText($, /your turns/)).toBe('your turns  paused on Haiku 5.5, which you picked · /vsr turns on')

  expect((await $.command.run({ command: 'vsr', args: 'turns' } as never)).text).toBe('vSR: your turns paused on Haiku 5.5, which you picked · /vsr turns on.')
  expect((await $.command.run({ command: 'vsr', args: 'turns on' } as never)).text).toBe('vSR: the router picks the model of your turns from now on.')
  await turn($, 't3', HARD, 1)
  expect(made.at(-1)?.model).toBe('opus')
})

test('a fallback the engine makes or a resumed model does not pause the router', TURNS, async ($, on) => {
  engine(on, { complexity: { score: 1.8, confidence: 0.9 } })
  on('classic.PostModelSwitch', () => ({}))
  session(on, { now: 'claude-sonnet-5-5' })
  const made = steps(on)

  await $.classic.PostModelSwitch(switched('claude-haiku-5-5', 'auto'))
  await $.classic.PostModelSwitch(switched('claude-haiku-5-5', 'resume'))
  await turn($, 't1', HARD, 1)

  expect(made[0]?.model).toBe('opus')
})

test('a turn that keeps its tier starts at the effort its prompt needs, a short reply at the last one', TURNS, async ($, on) => {
  const reply: Reply = { complexity: { score: 0.9, confidence: 0.9 }, steps: { score: 0.2, confidence: 0.8 } }
  engine(on, reply)
  session(on, { now: 'claude-sonnet-5-5' })
  const made = steps(on)

  await turn($, 't1', EASY, 1)
  reply.steps = { score: 1.4, confidence: 0.8 }
  await turn($, 't2', HARD.replace('Find', 'Explain'), 1)
  reply.steps = { score: 0.1, confidence: 0.9 }
  await turn($, 't3', YES, 1)

  // the session's own model is never named, only its effort set
  expect(made.map(one => [one.model, one.effort])).toEqual([
    ['claude-sonnet-5-5', 'low'],
    ['claude-sonnet-5-5', 'high'],
    ['claude-sonnet-5-5', 'high'],
  ])
})

test('a runtime that answers an error leaves the turn on its model and says so under the prompt', TURNS, async ($, on) => {
  engine(on, { errors: true })
  session(on, { now: 'claude-sonnet-5-5' })
  const made = steps(on)

  await turn($, 't1', HARD, 1)

  expect(made[0]?.model).toBe('claude-sonnet-5-5')
  const { texts } = await rowTree($, userRow(HARD))
  expect(texts.some(text => text.includes('turn ran on its model without routing'))).toBe(true)
})

test('a pushback moves the turn one step up from where the last ran, one effort level more', TURNS, async ($, on) => {
  const reply: Reply = { complexity: { score: 0.9, confidence: 0.9 }, steps: { score: 0.9, confidence: 0.8 } }
  engine(on, reply)
  session(on, { now: 'claude-sonnet-5-5' })
  const made = steps(on)

  await turn($, 't1', EASY, 1)
  reply.feedback = { SAT: 0.01, NEED_CLARIFICATION: 0.03, WRONG_ANSWER: 0.92, WANT_DIFFERENT: 0.03, NO_FEEDBACK: 0.01 }
  const pushback = 'No, that list is wrong: it skipped the nested folders. Walk the whole docs tree this time and print every file.'
  await turn($, 't2', pushback, 1)

  expect(made.map(one => [one.model, one.effort])).toEqual([
    ['claude-sonnet-5-5', 'medium'],
    ['opus', 'xhigh'],
  ])
  const { texts } = await rowTree($, userRow(pushback))
  expect(texts).toContain('↑  ESCALATED  you pushed back · this turn one step up')
  expect(texts).toContain('you pushed back: one step up, effort one level up → xhigh')
})

test('a prompt sent twice shows the newest route under it, not both turns stacked', TURNS, async ($, on) => {
  const reply: Reply = { complexity: { score: 1.8, confidence: 0.9 } }
  engine(on, reply)
  session(on, { now: 'claude-sonnet-5-5' })
  steps(on)

  await turn($, 't1', HARD, 1)
  reply.complexity = { score: 0.9, confidence: 0.9 }
  await turn($, 't2', EASY, 1)
  reply.complexity = { score: 1.8, confidence: 0.9 }
  await turn($, 't3', HARD, 1)

  const { texts } = await rowTree($, userRow(HARD))
  expect(texts.filter(text => text.startsWith('⇢  ROUTED'))).toEqual(['⇢  ROUTED  your turn → Opus · high effort'])
})

test('a session on a 1M window keeps the window on the model its turn moves to', TURNS, async ($, on) => {
  engine(on, { complexity: { score: 1.8, confidence: 0.9 } })
  session(on, { now: 'claude-sonnet-5-5[1m]' })
  const made = steps(on)

  await turn($, 't1', HARD, 1)

  expect(made[0]?.model).toBe('opus[1m]')
})

test('after /clear a move re-reads nothing, so the cache does not hold it', TURNS, async ($, on) => {
  const id = { now: 'session-1' }
  const reply: Reply = { complexity: { score: 1.8, confidence: 0.9 }, sessionId: id }
  engine(on, reply)
  session(on, { now: 'claude-opus-5-5' })
  const made = steps(on, { input_tokens: 500, cache_read_input_tokens: 100_000, output_tokens: 2000 })
  const last = { input_tokens: 500, output_tokens: 2000, cache_read_input_tokens: 100_000, cache_creation_input_tokens: 0, model: 'claude-opus-5-5' }

  await turn($, 't1', HARD, 1, last)
  id.now = 'session-2'
  reply.complexity = { score: 0.9, confidence: 0.95 }
  await turn($, 't2', EASY, 1)

  expect(made.at(-1)?.model).toBe('sonnet')
})

test('two prompts queued behind a running turn, the same text twice, each run on their own answer', TURNS, async ($, on) => {
  const reply: Reply = { complexity: { score: 1.8, confidence: 0.9 } }
  engine(on, reply)
  session(on, { now: 'claude-sonnet-5-5' })
  const made = steps(on)

  await submit($, HARD)
  await submit($, HARD)
  for (const turnId of ['t1', 't2']) {
    await $.turn.start({ turnId, text: HARD })
    for await (const _ of $.turn.step({ turnId, index: 0, model: 'claude-sonnet-5-5', effort: 'medium', messageCount: 1 })) void _
    await $.turn.complete({ turnId, answer: 'done' } as never)
  }

  expect(made.map(one => one.model)).toEqual(['opus', 'opus'])
})

test('a turn you did not type runs where the last one did', TURNS, async ($, on) => {
  engine(on, { complexity: { score: 1.8, confidence: 0.9 } })
  session(on, { now: 'claude-sonnet-5-5' })
  const made = steps(on)

  await turn($, 't1', HARD, 1)
  await turn($, 't2', '', 1)

  expect(made.map(one => one.model)).toEqual(['opus', 'opus'])
})

test('a routed turn counts in what routing came to, against the session model', TURNS, async ($, on) => {
  engine(on, { complexity: { score: 0.2, confidence: 0.95 } })
  session(on, { now: 'claude-opus-5-5' })
  steps(on)

  await turn($, 't1', EASY, 1, { input_tokens: 10_000, output_tokens: 1000, cache_read_input_tokens: 0, cache_creation_input_tokens: 0, model: 'claude-sonnet-5-5' })

  expect(await paneText($, /^routing/)).toBe('routing  $0.03 for 1 routed turn vs $0.06 on Opus 5.5 · saved 50% · est. at list price')
})

test('/vsr turns on asks for the router first when it is off', { options: { router: false } }, async ($, on) => {
  engine(on, {})

  expect((await $.command.run({ command: 'vsr', args: 'turns on' } as never)).text).toBe('vSR: turn on the router first: /vsr router on.')
})

// The plan alone, for the lines a live session rarely crosses.
const FACTS: TurnFacts = { level: 1, confidence: 0.9, steps: null, words: 20, session: 'claude-opus-5-5', now: undefined, isEscalated: false, context: null, last: null, lastEffort: undefined }
const SETTINGS = parseSettings({
  runtime: 'local-cpu', runtimeUrl: '', runtimeToken: '', runtimeModel: '', router: true, routerTurns: true, escalate: true, redact: true, shield: true, guard: true, check: false,
  redactWait: '1', shieldWait: '1', guardWait: '1', routerEasyBelow: 0.5, routerHardFrom: 1.5, routerMinConfidence: 0.3, routerDowngradeConfidence: 0.6,
  routerStepsLow: 0.5, routerStepsHigh: 1.2, routerStepsMinConfidence: 0.3, escalateFeedback: 0.8, escalateReask: 0.8, redactPii: 0.7, shieldInject: 0.9,
  guardReview: 0.9, checkUnsupported: 0.5, display: 'normal', routerGoal: 'balanced', logo: 'off', logPath: '', calibrateEvery: 0,
})

test('the plan keeps a smaller model away when unsure or with the quality goal, and above Opus a pushback has no step to climb', () => {
  expect(planTurn({ ...FACTS, confidence: 0.5 }, SETTINGS)).toMatchObject({ kind: 'keep', reason: 'downgrade', minConfidence: 0.6 })
  expect(planTurn(FACTS, { ...SETTINGS, routerGoal: 'quality' })).toMatchObject({ kind: 'keep', reason: 'quality' })
  expect(planTurn({ ...FACTS, confidence: 0.2 }, SETTINGS)).toMatchObject({ kind: 'keep', reason: 'unsure' })
  expect(planTurn({ ...FACTS, level: 1.9, session: 'claude-fable-5-1', isEscalated: true }, SETTINGS)).toEqual({ kind: 'stay' })
  expect(planTurn({ ...FACTS, level: 1.9, session: 'claude-fable-5-1' }, SETTINGS)).toMatchObject({ kind: 'move', model: 'opus' })
  expect(planTurn({ ...FACTS, level: 1.9 }, SETTINGS)).toEqual({ kind: 'stay' })
  // Haiku has no 1M window, so even the save goal keeps such a session's turns on Sonnet.
  expect(planTurn({ ...FACTS, level: 0.1, session: 'claude-opus-5-5[1m]' }, { ...SETTINGS, routerGoal: 'save' })).toMatchObject({ kind: 'move', model: 'sonnet' })
})
