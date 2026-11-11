import { expect, test } from 'claude-code/testing'

import { agentRow, engine, paneText, type Reply, rowTree, SPAWN, userRow } from './kit'

const FIRST = 'Summarize the tasks in TASKS.md in a subagent.'
const PUSHBACK = 'No, task 2 is wrong, it ignored the README. Redo it in a subagent.'
const THANKS = 'Thanks, that works perfectly.'
const WRONG = { SAT: 0.01, NEED_CLARIFICATION: 0.03, WRONG_ANSWER: 0.92, WANT_DIFFERENT: 0.03, NO_FEEDBACK: 0.01 }
const SATISFIED = { SAT: 0.96, NEED_CLARIFICATION: 0.01, WRONG_ANSWER: 0.01, WANT_DIFFERENT: 0.01, NO_FEEDBACK: 0.01 }
const NOTE = 'vSR: the user pushed back on your previous answer (they said it was wrong, 0.92). Re-check that answer before you build on it.'

type Test = Parameters<Parameters<typeof test>[1]>[0]
type Hooks = Parameters<Parameters<typeof test>[1]>[1]

function subagents(on: Hooks) {
  on('agent.spawn', (_, e) => ({ model: e.model ?? e.parentModel, agentId: `agent-${e.tool_use_id}` }))
  on('turn.step', async function* (_, e) {
    return { turnId: e.turnId, index: e.index, answer: '', toolUses: [] }
  })
}

// The engine stamps a typed prompt's origin; the kit's own `prompt.submit` does not.
const submit = ($: Test, text: string) => $.prompt.submit({ text, origin: { kind: 'composer' } } as never)

async function turn($: Test, turnId: string, text: string) {
  await submit($, text)
  await $.turn.start({ turnId, text })
}

const firstStepOf = async ($: Test, agentId: string) => {
  for await (const _ of $.turn.step({ turnId: 's1', index: 0, model: 'opus', effort: 'medium', messageCount: 1, agentId })) void _
}

test('a prompt that says the last answer was wrong tells Claude to re-check it, and its subagent goes one step up, to Opus at xhigh effort', { options: { display: 'verbose' } }, async ($, on) => {
  const reply: Reply = { feedback: WRONG, complexity: { score: 0.9, confidence: 0.8 }, steps: { score: 0.2, confidence: 0.8 }, contexts: [] }
  engine(on, reply)
  subagents(on)

  await turn($, 't1', PUSHBACK)
  const started = await $.agent.spawn(SPAWN)
  await firstStepOf($, 'agent-call-1')

  expect(started.model).toBe('opus')
  expect(reply.contexts).toEqual([[NOTE]])
  expect(await paneText($, /ESCALATED/, 100)).toBe(` ESCALATED  ${PUSHBACK}`)
  expect(await paneText($, /subagent → Opus/, 120)).toBe(
    'subagent → Opus, xhigh effort · multi-step task · difficulty 0.90 of 2 · confidence 0.80 · reasoning steps 0.20 · one level up → xhigh · escalated, router alone would pick Sonnet',
  )
  const { texts } = await rowTree($, agentRow('call-1'))
  expect(texts).toContain('⇢  ROUTED  subagent → Opus · xhigh effort')
  expect(texts).toContain('escalated: you pushed back · router alone would pick Sonnet')
})

test('an escalated subagent takes the steps rule on Opus, one level up: many steps reach max', { options: { display: 'verbose' } }, async ($, on) => {
  const reply: Reply = { feedback: WRONG, complexity: { score: 0.9, confidence: 0.8 }, steps: { score: 1.33, confidence: 0.35 } }
  engine(on, reply)
  subagents(on)

  await turn($, 't1', PUSHBACK)
  const started = await $.agent.spawn(SPAWN)
  await firstStepOf($, 'agent-call-1')
  const { texts } = await rowTree($, agentRow('call-1'))

  expect(started.model).toBe('opus')
  expect(texts).toContain('⇢  ROUTED  subagent → Opus · max effort')
  expect(await paneText($, /subagent → Opus/, 160)).toContain('steps 1.33 · one level up → max · escalated, router alone would pick Sonnet')
})

test('an escalated subagent whose steps answer is unsure goes one level above the default, to xhigh', { options: { display: 'verbose' } }, async ($, on) => {
  const reply: Reply = { feedback: WRONG, complexity: { score: 0.2, confidence: 0.1 }, steps: { score: 0.1, confidence: 0.1 } }
  engine(on, reply)
  subagents(on)

  await turn($, 't1', PUSHBACK)
  await $.agent.spawn(SPAWN)
  await firstStepOf($, 'agent-call-1')
  const { texts } = await rowTree($, agentRow('call-1'))

  expect(texts).toContain('⇢  ROUTED  subagent → Opus · xhigh effort')
  expect(await paneText($, /subagent → Opus/, 160)).toContain('steps 0.10 · one level up → xhigh · escalated, router alone would keep the default model')
})

test('an escalated subagent keeps a higher effort it would have had', { options: { display: 'verbose' } }, async ($, on) => {
  const reply: Reply = { feedback: WRONG, complexity: { score: 0.9, confidence: 0.8 }, steps: { score: 0.2, confidence: 0.8 } }
  engine(on, reply)
  subagents(on)

  await turn($, 't1', PUSHBACK)
  await $.agent.spawn(SPAWN)
  for await (const _ of $.turn.step({ turnId: 's1', index: 0, model: 'opus', effort: 'max', messageCount: 1, agentId: 'agent-call-1' })) void _

  expect((await rowTree($, agentRow('call-1'))).texts).toContain('⇢  ROUTED  subagent → Opus · max effort')
})

test('an escalated subagent still goes one step up when the classifier fails, to Opus at xhigh', async ($, on) => {
  const reply: Reply = { feedback: WRONG }
  engine(on, reply)
  on('agent.spawn', (_, e) => ({ model: e.model ?? e.parentModel, agentId: 'agent-call-1' }))
  const efforts: unknown[] = []
  on('turn.step', async function* (_, e) {
    efforts.push(e.effort)
    return { turnId: e.turnId, index: e.index, answer: '', toolUses: [] }
  })

  await turn($, 't1', PUSHBACK)
  reply.fails = true
  const started = await $.agent.spawn(SPAWN)
  await firstStepOf($, 'agent-call-1')

  expect(started.model).toBe('opus')
  expect(efforts).toEqual(['xhigh'])
})

test('a satisfied prompt escalates nothing, and its subagent keeps the router pick', async ($, on) => {
  const reply: Reply = { feedback: SATISFIED, complexity: { score: 0.9, confidence: 0.8 }, contexts: [] }
  engine(on, reply)
  subagents(on)

  await turn($, 't1', THANKS)
  const started = await $.agent.spawn(SPAWN)

  expect(started.model).toBe('sonnet')
  expect(reply.contexts).toEqual([undefined])
  expect(await paneText($, /ESCALATED|escalated/)).toBeUndefined()
})

test('the escalation ends with the turn that answers the pushback', async ($, on) => {
  const reply: Reply = { feedback: WRONG, complexity: { score: 0.9, confidence: 0.8 } }
  engine(on, reply)
  subagents(on)

  await turn($, 't1', PUSHBACK)
  await $.turn.complete({ turnId: 't1', answer: 'Redone.', reason: 'answer', durationMs: 1, isAborted: false })
  const started = await $.agent.spawn(SPAWN)

  expect(started.model).toBe('sonnet')
})

test('a prompt that repeats the one before escalates on the repeat alone', { options: { display: 'verbose' } }, async ($, on) => {
  const reply: Reply = { feedback: SATISFIED, reask: 0.83, bodies: [] }
  engine(on, reply)

  await submit($, FIRST)
  expect(reply.bodies?.filter(body => body.questions.reask)).toEqual([])
  await submit($, 'Summarize TASKS.md again, in a subagent.')

  expect(reply.bodies?.find(body => body.questions.reask)?.state).toEqual({
    current: 'Summarize TASKS.md again, in a subagent.',
    prior: FIRST,
  })
  const { texts } = await rowTree($, userRow('Summarize TASKS.md again, in a subagent.'))
  expect(texts).toContain('asked again 0.83')
})

test('the ESCALATED card is drawn under the prompt it judged, and no other', { options: { display: 'verbose' } }, async ($, on) => {
  engine(on, { feedback: WRONG })

  await submit($, PUSHBACK)
  const { texts, cards } = await rowTree($, userRow(PUSHBACK))
  const other = await rowTree($, userRow(THANKS))

  expect(cards).toEqual([expect.objectContaining({ borderStyle: 'round', borderColor: 'claude', paddingX: 1, width: 76 })])
  expect(texts).toContain('↑  ESCALATED  you pushed back · next subagent one step up')
  expect(texts).toContain(' vSR  · escalate')
  expect(texts).toContain('said it was wrong 0.92')
  expect(texts).toContain('told Claude to re-check its last answer')
  expect(other.cards).toEqual([])
})

test('a classifier that fails escalates nothing and shows SKIPPED under the prompt', { options: { display: 'verbose' } }, async ($, on) => {
  const reply: Reply = { errors: true, contexts: [] }
  engine(on, reply)
  subagents(on)

  await turn($, 't1', PUSHBACK)
  const started = await $.agent.spawn(SPAWN)
  const { texts, cards } = await rowTree($, userRow(PUSHBACK))

  expect(started.model).toBe('opus')
  expect(reply.contexts).toEqual([undefined])
  expect(cards).toEqual([expect.objectContaining({ borderColor: 'inactive' })])
  expect(texts).toContain('○  SKIPPED  pushback not checked')
})
