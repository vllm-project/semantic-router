import { expect, test } from 'claude-code/testing'

import {
  agentRow,
  ENGINE_ROW,
  engine,
  logLines,
  paneText,
  type Reply,
  rowTree,
  SPAWN,
  type Test,
} from './kit'

const COMPLEXITY_QUESTION = {
  type: 'score',
  instructions: 'How difficult is this request to solve correctly?',
  criteria: ['straightforward retrieval or formatting', 'several steps of reasoning', 'advanced reasoning or specialized expertise'],
}
const STEPS_QUESTION = {
  type: 'score',
  instructions: 'How many reasoning steps does this task need?',
  criteria: ['one or two steps', 'several steps that build on each other', 'many steps or long careful reasoning'],
}

function subagent(on: Parameters<typeof engine>[0]) {
  on('agent.spawn', (_, e) => ({ model: e.model ?? e.parentModel, agentId: 'agent-1' }))
  on('turn.step', async function* (_, e) {
    return { turnId: e.turnId, index: e.index, answer: '', toolUses: [] }
  })
}

// The subagent's first request, which it would have made at `effort`.
async function firstStep($: Test, effort: string) {
  const step = $.turn.step({ turnId: 't1', index: 0, model: 'haiku', effort, messageCount: 1, agentId: 'agent-1' })
  for await (const _ of step) void _
}

const logged = async (clock: ReturnType<typeof engine>, reply: Reply) => {
  await clock.advance(0)
  return logLines(reply)[0].detail
}

test('a routed subagent shows a ROUTED entry, with the effort its first request ran at', async ($, on) => {
  engine(on, { complexity: { score: 0.09, confidence: 0.86 }, steps: { score: 0.17, confidence: 0.74 } })
  subagent(on)

  const started = await $.agent.spawn(SPAWN)
  expect(started.model).toBe('haiku')
  expect(await paneText($, /subagent →/, 120)).toBe(
    'subagent → Haiku · easy task · difficulty 0.09 of 2 · confidence 0.86 · reasoning steps 0.17 · Haiku runs at medium',
  )

  await firstStep($, 'high')
  expect(await paneText($, /subagent →/, 120)).toContain('subagent → Haiku, medium effort')
  expect(await paneText($, /ROUTED/)).toBe(' ROUTED  Rename a variable')
})

test('the router asks complexity and steps in one request, each question as decided', async ($, on) => {
  const reply: Reply = { complexity: { score: 0.09, confidence: 0.86 }, steps: { score: 0.17, confidence: 0.74 }, bodies: [] }
  engine(on, reply)
  subagent(on)

  await $.agent.spawn(SPAWN)

  expect(reply.bodies).toEqual([
    { state: SPAWN.prompt, questions: { complexity: COMPLEXITY_QUESTION, steps: STEPS_QUESTION } },
  ])
})

// The measured tasks of the demo: complexity and steps as the live runtime answered them.
const MEASURED = [
  { task: 'List the files in this repo.', complexity: [0.09, 0.86], steps: [0.17, 0.74], model: 'haiku', effort: 'medium' },
  { task: 'the timing-proof task', complexity: [1.61, 0.42], steps: [1.33, 0.35], model: 'opus', effort: 'xhigh' },
] as const

for (const { task, complexity, steps, model, effort } of MEASURED) {
  test(`${task} routes to ${model} at ${effort} effort`, async ($, on) => {
    engine(on, { complexity: { score: complexity[0], confidence: complexity[1] }, steps: { score: steps[0], confidence: steps[1] } })
    subagent(on)

    const started = await $.agent.spawn(SPAWN)
    await firstStep($, 'medium')

    expect(started.model).toBe(model)
    expect(await paneText($, /subagent →/, 140)).toContain(`subagent → ${model[0].toUpperCase()}${model.slice(1)}, ${effort} effort`)
  })
}

// The refactor task as measured: confidence 0.39, past the router's line but short of a smaller model's.
const REFACTOR = { complexity: { score: 0.62, confidence: 0.39 }, steps: { score: 0.52, confidence: 0.23 } }

test("the refactor task stays on the session's Opus: 0.39 is too unsure for a smaller model", async ($, on) => {
  engine(on, REFACTOR)
  subagent(on)

  const started = await $.agent.spawn(SPAWN)

  expect(started.model).toBe('opus')
  expect(await paneText($, /smaller model/, 140)).toBe('subagent stays on Opus · not sure enough for a smaller model · confidence 0.39 < 0.60')
})

test('with the goal save the refactor task goes to Sonnet', { options: { routerGoal: 'save' } }, async ($, on) => {
  engine(on, REFACTOR)
  subagent(on)

  expect((await $.agent.spawn(SPAWN)).model).toBe('sonnet')
})

test("with the goal quality even an easy task keeps the session's model", { options: { routerGoal: 'quality' } }, async ($, on) => {
  engine(on, { complexity: { score: 0.09, confidence: 0.86 }, steps: { score: 0.17, confidence: 0.74 } })
  subagent(on)

  expect((await $.agent.spawn(SPAWN)).model).toBe('opus')
  expect(await paneText($, /smaller model/, 140)).toBe('subagent stays on Opus · the routing goal is quality: never a smaller model')
})

test('a hard task goes to Opus when the session runs Sonnet: a bigger model needs only the router line', async ($, on) => {
  engine(on, { complexity: { score: 1.61, confidence: 0.42 }, steps: { score: 1.33, confidence: 0.35 } })
  subagent(on)

  expect((await $.agent.spawn({ ...SPAWN, parentModel: 'sonnet' })).model).toBe('opus')
})

// Every cell of the effort table, by the steps score against the 0.5 and 1.2 lines.
const CELLS = [
  { model: 'haiku', complexity: 0.1, steps: 0.2, effort: 'medium' },
  { model: 'haiku', complexity: 0.1, steps: 0.9, effort: 'medium' },
  { model: 'haiku', complexity: 0.1, steps: 1.6, effort: 'medium' },
  { model: 'sonnet', complexity: 0.9, steps: 0.2, effort: 'low' },
  { model: 'sonnet', complexity: 0.9, steps: 0.9, effort: 'medium' },
  { model: 'sonnet', complexity: 0.9, steps: 1.6, effort: 'high' },
  { model: 'opus', complexity: 1.7, steps: 0.2, effort: 'high' },
  { model: 'opus', complexity: 1.7, steps: 0.9, effort: 'high' },
  { model: 'opus', complexity: 1.7, steps: 1.6, effort: 'xhigh' },
] as const

for (const { model, complexity, steps, effort } of CELLS) {
  test(`${model} with steps ${steps} runs at ${effort} effort`, async ($, on) => {
    const reply: Reply = { complexity: { score: complexity, confidence: 0.8 }, steps: { score: steps, confidence: 0.8 }, files: {} }
    const clock = engine(on, reply)
    subagent(on)

    await $.agent.spawn(SPAWN)
    await firstStep($, 'medium')

    expect(await logged(clock, reply)).toMatchObject({ model, effort, steps: { score: steps, confidence: 0.8 } })
  })
}

test('a steps score on a line takes the higher rung', async ($, on) => {
  const reply: Reply = { complexity: { score: 0.9, confidence: 0.8 }, steps: { score: 0.5, confidence: 0.8 }, files: {} }
  const clock = engine(on, reply)
  subagent(on)

  await $.agent.spawn(SPAWN)
  await firstStep($, 'high')

  expect(await logged(clock, reply)).toMatchObject({ model: 'sonnet', effort: 'medium', effortFrom: 'steps' })
})

test('an unsure steps answer gives the band its default effort, whatever the score says', async ($, on) => {
  const reply: Reply = { complexity: { score: 0.9, confidence: 0.8 }, steps: { score: 1.9, confidence: 0.29 }, files: {} }
  const clock = engine(on, reply)
  subagent(on)

  await $.agent.spawn(SPAWN)
  await firstStep($, 'low')

  expect(await logged(clock, reply)).toMatchObject({ model: 'sonnet', effort: 'medium', steps: { score: 1.9, confidence: 0.29 }, effortFrom: 'unsure' })
  expect(await paneText($, /subagent →/, 140)).toContain('steps unsure (0.29) → default medium')
})

test('an unsure steps answer sends Opus to high, not xhigh', async ($, on) => {
  const reply: Reply = { complexity: { score: 1.61, confidence: 0.42 }, steps: { score: 1.9, confidence: 0.1 }, files: {} }
  const clock = engine(on, reply)
  subagent(on)

  await $.agent.spawn(SPAWN)
  await firstStep($, 'medium')

  expect(await logged(clock, reply)).toMatchObject({ model: 'opus', effort: 'high', effortFrom: 'unsure' })
})

test('a steps answer that failed still routes by complexity, at the band default effort', { options: { display: 'verbose' } }, async ($, on) => {
  const reply: Reply = { complexity: { score: 0.9, confidence: 0.8 }, steps: 'fails', files: {} }
  const clock = engine(on, reply)
  subagent(on)

  const started = await $.agent.spawn(SPAWN)
  await firstStep($, 'high')
  const { texts, cards } = await rowTree($, agentRow('call-1'))

  expect(started.model).toBe('sonnet')
  expect(await logged(clock, reply)).toMatchObject({ model: 'sonnet', effort: 'medium', steps: null, effortFrom: 'unsure' })
  expect(cards).toEqual([expect.objectContaining({ borderColor: 'suggestion' })])
  expect(texts).toContainEqual('reasoning steps unavailable → default medium · Rename a variable')
  expect(texts).not.toContainEqual(expect.stringContaining('SKIPPED'))
})

test('a steps answer without a score is a failed one', async ($, on) => {
  const reply: Reply = { answers: { complexity: { type: 'score', score: 0.9, confidence: 0.8 }, steps: { type: 'score', score: 'many' } } }
  engine(on, reply)
  subagent(on)

  const started = await $.agent.spawn(SPAWN)

  expect(started.model).toBe('sonnet')
})

test('an answer with no complexity is SKIPPED even when steps answered', { options: { display: 'verbose' } }, async ($, on) => {
  engine(on, { answers: { steps: { type: 'score', score: 0.9, confidence: 0.8 } } })
  on('agent.spawn', () => ({ model: 'opus', agentId: 'agent-1' }))

  const started = await $.agent.spawn(SPAWN)
  const { texts } = await rowTree($, agentRow('call-1'))

  expect(started.model).toBe('opus')
  expect(texts).toContain('○  SKIPPED  subagent started without routing')
})

test('Opus never runs below the effort the subagent would have had', async ($, on) => {
  const reply: Reply = { complexity: { score: 1.61, confidence: 0.42 }, steps: { score: 0.1, confidence: 0.8 }, files: {} }
  const clock = engine(on, reply)
  subagent(on)

  await $.agent.spawn(SPAWN)
  await firstStep($, 'max')

  expect(await logged(clock, reply)).toMatchObject({ model: 'opus', effort: 'max', effortFrom: 'floor' })
  expect(await paneText($, /subagent →/, 160)).toContain('steps 0.10 · keeps its own max effort')
})

test('Opus raises a lower effort to what steps picks, and a higher one wins only when it is above', async ($, on) => {
  const reply: Reply = { complexity: { score: 1.61, confidence: 0.42 }, steps: { score: 1.33, confidence: 0.35 }, files: {} }
  const clock = engine(on, reply)
  subagent(on)

  await $.agent.spawn(SPAWN)
  await firstStep($, 'xhigh')

  expect(await logged(clock, reply)).toMatchObject({ model: 'opus', effort: 'xhigh', effortFrom: 'steps' })
})

test('Sonnet has no floor: steps lowers an effort the subagent would have had', async ($, on) => {
  const reply: Reply = { complexity: { score: 0.9, confidence: 0.8 }, steps: { score: 0.2, confidence: 0.8 }, files: {} }
  const clock = engine(on, reply)
  subagent(on)

  await $.agent.spawn(SPAWN)
  await firstStep($, 'xhigh')

  expect(await logged(clock, reply)).toMatchObject({ model: 'sonnet', effort: 'low', effortFrom: 'steps' })
})

test('the steps lines follow their options', {
  options: { routerStepsLow: 0.1, routerStepsHigh: 0.2, routerStepsMinConfidence: 0.5 },
}, async ($, on) => {
  const reply: Reply = { complexity: { score: 0.9, confidence: 0.8 }, steps: { score: 0.3, confidence: 0.4 }, files: {} }
  const clock = engine(on, reply)
  subagent(on)

  await $.agent.spawn(SPAWN)
  await firstStep($, 'medium')

  expect(await logged(clock, reply)).toMatchObject({ model: 'sonnet', effort: 'medium', effortFrom: 'unsure' })
})

test('the steps lines move a pick', {
  options: { routerStepsLow: 0.1, routerStepsHigh: 0.2 },
}, async ($, on) => {
  const reply: Reply = { complexity: { score: 0.9, confidence: 0.8 }, steps: { score: 0.3, confidence: 0.8 }, files: {} }
  const clock = engine(on, reply)
  subagent(on)

  await $.agent.spawn(SPAWN)
  await firstStep($, 'medium')

  expect(await logged(clock, reply)).toMatchObject({ model: 'sonnet', effort: 'high', effortFrom: 'steps' })
})

test('a vague answer keeps the subagent on its default and shows KEPT', async ($, on) => {
  engine(on, { complexity: { score: 1, confidence: 0.1 } })
  on('agent.spawn', () => ({ model: 'claude-opus-5-5', agentId: 'agent-2' }))

  const started = await $.agent.spawn(SPAWN)

  expect(started.model).toBe('claude-opus-5-5')
  expect(await paneText($, /KEPT/)).toBe(' KEPT  Rename a variable')
  expect(await paneText($, /stays on/)).toBe('subagent stays on Opus 5.5 · model unsure · confidence 0.10 < 0.30')
})

test('a kept subagent says so in one framed line under its row, so the user sees the router answered', async ($, on) => {
  engine(on, { complexity: { score: 1, confidence: 0.1 } })
  on('agent.spawn', () => ({ model: 'claude-opus-5-5', agentId: 'agent-2' }))
  await $.agent.spawn(SPAWN)

  const { texts, cards, frames } = await rowTree($, agentRow('call-1'))

  expect(cards).toEqual([])
  expect(frames).toEqual([expect.objectContaining({ borderStyle: 'round', borderColor: 'suggestion', borderDimColor: true })])
  expect(texts).toContain(' vSR   KEPT  subagent stays on Opus 5.5 · model unsure')
})

test('a row too narrow for a frame keeps the one-line form unframed', async ($, on) => {
  engine(on, { complexity: { score: 1, confidence: 0.1 } })
  on('agent.spawn', () => ({ model: 'claude-opus-5-5', agentId: 'agent-2' }))
  await $.agent.spawn(SPAWN)

  const { texts, frames } = await rowTree($, agentRow('call-1', 26))

  expect(frames).toEqual([])
  expect(texts.some(text => text.startsWith(' vSR   KEPT'))).toBe(true)
})

test('an Agent row with a route draws a ROUTED card with the difficulty meter, and the effort once known', { options: { display: 'verbose' } }, async ($, on) => {
  engine(on, { complexity: { score: 0.26, confidence: 0.61 }, steps: { score: 0.17, confidence: 0.74 } })
  subagent(on)
  await $.agent.spawn(SPAWN)

  const before = await rowTree($, agentRow('call-1'))
  expect(before.texts[0]).toBe(ENGINE_ROW)
  expect(before.cards).toEqual([expect.objectContaining({ borderStyle: 'round', borderColor: 'suggestion' })])
  expect(before.texts).toContain('⇢  ROUTED  subagent → Haiku')

  await firstStep($, 'high')
  const { texts } = await rowTree($, agentRow('call-1'))

  expect(texts).toContain('⇢  ROUTED  subagent → Haiku · medium effort')
  expect(texts).toContain('easy ▕███▏░░│░░░░░░░░░░░░│░░░░░░▏ hard  0.26 of 2 · easy task')
  expect(texts).toContainEqual('reasoning steps 0.17 · Haiku runs at medium · Rename a variable')
})

test('the ROUTED footnote says where the effort came from', { options: { display: 'verbose', routerGoal: 'save' } }, async ($, on) => {
  const reply: Reply = { complexity: { score: 1.61, confidence: 0.42 }, steps: { score: 1.33, confidence: 0.35 } }
  engine(on, reply)
  on('agent.spawn', (_, e) => ({ model: e.model ?? e.parentModel, agentId: e.tool_use_id }))
  const noteOf = async (id: string) => (await rowTree($, agentRow(id))).texts.find(text => text.endsWith('· Fix a typo'))
  const spawn = (id: string) => $.agent.spawn({ ...SPAWN, tool_use_id: id, description: 'Fix a typo' })

  await spawn('a')
  reply.complexity = { score: 0.62, confidence: 0.39 }
  reply.steps = { score: 0.52, confidence: 0.23 }
  await spawn('b')
  reply.steps = 'fails'
  await spawn('c')
  reply.complexity = { score: 0.09, confidence: 0.86 }
  reply.steps = { score: 0.17, confidence: 0.74 }
  await spawn('d')
  reply.steps = 'fails'
  await spawn('e')

  expect(await noteOf('a')).toBe('reasoning steps 1.33 → xhigh · Fix a typo')
  expect(await noteOf('b')).toBe('reasoning steps unsure (0.23) → default medium · Fix a typo')
  expect(await noteOf('c')).toBe('reasoning steps unavailable → default medium · Fix a typo')
  expect(await noteOf('d')).toBe('reasoning steps 0.17 · Haiku runs at medium · Fix a typo')
  expect(await noteOf('e')).toBe('Haiku runs at medium · Fix a typo')
})

test('a footnote longer than the card is clipped to it, and the card keeps its width', { options: { display: 'verbose', routerGoal: 'save' } }, async ($, on) => {
  engine(on, { complexity: { score: 0.62, confidence: 0.39 }, steps: { score: 0.52, confidence: 0.23 } })
  subagent(on)
  await $.agent.spawn(SPAWN)

  const { texts, cards } = await rowTree($, agentRow('call-1', 60))

  expect(cards).toEqual([expect.objectContaining({ width: 52 })])
  // The lines take the card's 52 columns but its border and padding.
  expect(texts).toContain('reasoning steps unsure (0.23) → default medium …')
})

test('the one-line form of a route keeps its effort, and gives up the detail before it', async ($, on) => {
  engine(on, { complexity: { score: 0.09, confidence: 0.86 }, steps: { score: 0.17, confidence: 0.74 } })
  subagent(on)
  await $.agent.spawn(SPAWN)
  await firstStep($, 'high')

  const { texts, cards } = await rowTree($, agentRow('call-1', 55))

  expect(cards).toEqual([])
  expect(texts).toContain(' vSR   ROUTED  subagent → Haiku, medium effort')
})

test('a vague answer draws a KEPT card with the difficulty meter', { options: { display: 'verbose' } }, async ($, on) => {
  engine(on, { complexity: { score: 1, confidence: 0.27 } })
  on('agent.spawn', () => ({ model: 'claude-opus-5-5', agentId: 'agent-2' }))
  await $.agent.spawn(SPAWN)

  const { texts, cards } = await rowTree($, agentRow('call-1'))

  expect(cards).toEqual([expect.objectContaining({ borderColor: 'suggestion' })])
  expect(texts).toContain('≈  KEPT  subagent stays on Opus 5.5')
  expect(texts).toContain('easy ▕██████│██████░░░░░░│░░░░░░▏ hard  1.00 of 2 · multi-step task')
  expect(texts).toContain('model unsure · confidence 0.27 < 0.30')
})

test('an Agent row whose classifier failed draws a SKIPPED card of two lines and no meter', { options: { display: 'verbose' } }, async ($, on) => {
  engine(on, { errors: true })
  on('agent.spawn', () => ({ model: 'opus', agentId: 'agent-1' }))
  await $.agent.spawn(SPAWN)

  const { texts, cards } = await rowTree($, agentRow('call-1'))

  expect(cards).toEqual([expect.objectContaining({ borderColor: 'inactive' })])
  expect(texts).toContain('○  SKIPPED  subagent started without routing')
  expect(texts).toContainEqual('the decision model answered with an error (HTTP 500)')
  expect(texts.filter(text => text.includes('▕'))).toEqual([])
})

test('a resumed subagent keeps the model and effort it ran on: nothing is asked and nothing is recorded', async ($, on) => {
  const reply: Reply = { complexity: { score: 1.8, confidence: 0.95 }, steps: { score: 1.8, confidence: 0.95 }, bodies: [] }
  engine(on, reply)
  subagent(on)

  const started = await $.agent.spawn({ ...SPAWN, resume: { agentId: 'agent-1' } } as never)

  expect(started.model).toBe('opus')
  expect(reply.bodies).toEqual([])
  expect(await paneText($, /ROUTED|KEPT/)).toBeUndefined()
})

test('a route to the session\'s own tier names no model, so a pinned session model is kept', async ($, on) => {
  engine(on, { complexity: { score: 1.8, confidence: 0.95 }, steps: { score: 1.8, confidence: 0.95 } })
  subagent(on)

  const started = await $.agent.spawn({ ...SPAWN, parentModel: 'claude-opus-4-8' })

  expect(started.model).toBe('claude-opus-4-8')
})
