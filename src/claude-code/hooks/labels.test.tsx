import { expect, test } from 'claude-code/testing'

import { HOME, isControl, PANE, PUSH, type Reply, SPAWN, engine, paneLines } from './kit'
import { costOf } from './lib/prices'

type Test = Parameters<Parameters<typeof test>[1]>[0]

const LABELS = `${HOME}/.claude/vsr/labels.jsonl`

async function pane($: Test) {
  return $.ui.mount({ ...PANE, surface: 'terminal', viewport: { columns: 80, rows: 60 } })
}

async function buttonsIn($: Test) {
  const ui = await pane($)
  const keys = (await ui.findAll({ type: 'Button' })).filter(one => !isControl(one.key)).map(one => String(one.props.key))
  await ui.unmount()
  return keys
}

const labelLines = (reply: Reply) => (reply.files?.[LABELS] ?? '').trim().split('\n').filter(Boolean).map(line => JSON.parse(line))

test('saying an allowed command was the wrong call labels it one to hold, beside the log, and marks the answer', async ($, on) => {
  const reply: Reply = { review: 0.2, files: {} }
  engine(on, reply)
  on('tool.call', () => ({ result: { stdout: '' } }))
  await $.tool.call({ ...PUSH, command: 'git status', tool_use_id: 'g1' })

  const ui = await pane($)
  await ui.press({ key: 'g1:wrong' })
  await ui.unmount()
  const again = await pane($)
  const chosen = (await again.findAll({ type: 'Button' })).find(one => one.props.key === 'g1:wrong')?.props
  await again.unmount()

  expect(labelLines(reply)).toEqual([expect.objectContaining({ id: 'g1', app: 'guard', label: 'yes' })])
  expect(chosen).toMatchObject({ label: '✓ wrong call', variant: 'primary' })
  expect(chosen).toMatchObject({ plain: true })
})

test('an allowed command takes two answers, right call and wrong call', async ($, on) => {
  engine(on, { review: 0.2, files: {} })
  on('tool.call', () => ({ result: { stdout: '' } }))
  await $.tool.call({ ...PUSH, command: 'git status', tool_use_id: 'g1' })

  expect(await buttonsIn($)).toEqual(['g1:right', 'g1:wrong'])
})

test("a route's answer is the model it needed, as /vsr:calibrate reads it", async ($, on) => {
  const reply: Reply = { complexity: { score: 0.09, confidence: 0.86 }, steps: { score: 0.17, confidence: 0.74 }, files: {} }
  engine(on, reply)
  on('agent.spawn', (_, e) => ({ model: e.model ?? e.parentModel, agentId: 'agent-1' }))
  await $.agent.spawn(SPAWN)

  const ui = await pane($)
  await ui.press({ key: 'agent-1:too-low' })
  await ui.unmount()

  expect(labelLines(reply)).toEqual([expect.objectContaining({ id: 'agent-1', app: 'router', label: 'too-low' })])
})

test('a command held by a fixed rule takes no answer: its score did not decide', async ($, on) => {
  engine(on, { review: 0.1, files: {} })
  on('classic.PermissionRequest', () => ({ decision: 'deny' }) as never)
  await $.tool.check({ tool: 'Bash', input: { command: 'rm -rf ~' }, tool_use_id: 'r1' } as never)

  expect(await buttonsIn($)).toEqual([])
})

test('with the log off the pane offers no answers, since nothing would keep them', { options: { logPath: '' } }, async ($, on) => {
  engine(on, { review: 0.2 })
  on('tool.call', () => ({ result: { stdout: '' } }))
  await $.tool.call({ ...PUSH, command: 'git status', tool_use_id: 'g1' })

  expect(await buttonsIn($)).toEqual([])
})

const USAGE = { model: 'claude-haiku-5-5', input_tokens: 1_000_000, output_tokens: 100_000, cache_read_input_tokens: 0, cache_creation_input_tokens: 0 }

async function routedTurn($: Test, usage: typeof USAGE) {
  await $.turn.complete({ turnId: 's1', answer: '', reason: 'answer', durationMs: 1, isAborted: false, agentId: 'agent-1', usage })
}

test("the pane says what a routed subagent cost against the session's model, at list price", async ($, on) => {
  engine(on, { complexity: { score: 0.09, confidence: 0.86 }, steps: { score: 0.17, confidence: 0.74 } })
  on('agent.spawn', (_, e) => ({ model: e.model ?? e.parentModel, agentId: 'agent-1' }))
  await $.agent.spawn(SPAWN)
  await routedTurn($, USAGE)

  const routing = (await paneLines($, { columns: 140, rows: 60 })).find(line => line.startsWith('routing'))

  expect(routing).toBe('routing  $0.15 for 1 subagent turn vs $6.00 on Opus · saved 98% · est. at list price')
})

test('the spend of a long subagent counts though its route is older than the pane\'s last hundred decisions', async ($, on) => {
  engine(on, { complexity: { score: 0.09, confidence: 0.86 }, steps: { score: 0.17, confidence: 0.74 }, review: 0.1 })
  on('agent.spawn', (_, e) => ({ model: e.model ?? e.parentModel, agentId: 'agent-1' }))
  await $.agent.spawn(SPAWN)
  for (let i = 0; i < 120; i += 1) await $.tool.check({ tool: 'Bash', input: { command: `make test${i}` }, tool_use_id: `c${i}` } as never)
  await routedTurn($, USAGE)

  const routing = (await paneLines($, { columns: 140, rows: 60 })).find(line => line.startsWith('routing'))

  expect(routing).toBe('routing  $0.15 for 1 subagent turn vs $6.00 on Opus · saved 98% · est. at list price')
})

test('a subagent sent to a stronger model than the session shows what it cost more', async ($, on) => {
  engine(on, { complexity: { score: 1.7, confidence: 0.9 }, steps: { score: 1.6, confidence: 0.9 } })
  on('agent.spawn', (_, e) => ({ model: e.model ?? e.parentModel, agentId: 'agent-1' }))
  await $.agent.spawn({ ...SPAWN, parentModel: 'claude-sonnet-5-5' })
  await routedTurn($, { ...USAGE, model: 'claude-opus-5-5' })

  const routing = (await paneLines($, { columns: 140, rows: 60 })).find(line => line.startsWith('routing'))

  expect(routing).toBe('routing  $6.00 for 1 subagent turn vs $3.00 on Sonnet 5.5 · cost 100% more · est. at list price')
})

test('a cache write costs 1.25 times the input price, a model with no known price costs nothing known', () => {
  expect(costOf('claude-opus-5-5', { ...USAGE, input_tokens: 0, output_tokens: 0, cache_creation_input_tokens: 1_000_000 })).toBe(5)
  expect(costOf('some-other-model', USAGE)).toBeUndefined()
})

test('Fable 5 cache reads cost four times Fable 5.1\'s, and a retired Opus 4 has no price', () => {
  const reads = { ...USAGE, input_tokens: 0, output_tokens: 0, cache_read_input_tokens: 1_000_000 }

  expect(costOf('claude-fable-5', reads)).toBe(1)
  expect(costOf('claude-fable-5-1', reads)).toBe(0.25)
  expect(costOf('claude-opus-4-20250514', USAGE)).toBeUndefined()
  expect(costOf('claude-opus-4-1', USAGE)).toBeUndefined()
  expect(costOf('claude-opus-4-8', USAGE)).toBe(7.5)
})

test('an answer not given yet is underlined, as a link is, in its dim colour', async ($, on) => {
  engine(on, { review: 0.2, files: {} })
  on('tool.call', () => ({ result: { stdout: '' } }))
  await $.tool.call({ ...PUSH, command: 'git status', tool_use_id: 'g1' })

  const ui = await pane($)
  const texts = await ui.findAll({ type: 'Text' })
  await ui.unmount()
  const label = texts.find(one => one.text === 'wrong call')?.props

  expect(label).toMatchObject({ underline: true, dimColor: true })
})
