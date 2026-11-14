import { expect, test } from 'claude-code/testing'
import type { ConfigRow, On } from 'claude-code'

import { engine, type Reply, type Test } from './kit'
import { listingOf, profileOf } from './lib/profile'
import { parseSettings } from './lib/settings'

const REMOTE = { runtime: 'remote', runtimeUrl: 'http://localhost:8899/v1/decisions', runtimeModel: 'vela3-lite' } as const
const START = { cwd: '/', surface: 'terminal', isInteractive: true } as const
const PII = 'Ticket from Maria Gonzalez <maria.gonzalez@example.com>'

const readOf = (content: string) => ({ result: { type: 'text', file: { filePath: '/repo/t.txt', content, numLines: 1, startLine: 1, totalLines: 1 } } })
const contentOf = (ran: { result?: unknown }) => (ran.result as { file: { content: string } }).file.content

// The engine's Read: a file with a name and an address.
const files = (on: On) => on('tool.call', { tool: 'Read' }, () => readOf(PII) as never)
const readPii = async ($: Test) => contentOf(await $.tool.call({ tool: 'Read', tool_use_id: 'r1', input: { file_path: '/repo/t.txt' } } as never))

// The rows `/config` draws for the options a test changes, and every change made through it.
function configRows(on: On) {
  const sets: { key: string; value: unknown }[] = []
  const row = (key: string, value: ConfigRow['value']) => ({ key, label: key, kind: 'toggle', value, provider: { kind: 'plugin', name: 'vsr' }, isLocked: false })
  on('config.list', () => ({ value: [row('vsr.redact', true), row('vsr.guard', true)] as never }))
  on('config.set', (_, e) => {
    sets.push({ key: e.key, value: e.value })
    return { value: e.value }
  })
  return sets
}

const SPANS = [
  { label: 'PERSON', start: 0, end: 0, text: 'Maria Gonzalez', probability: 0.95 },
  { label: 'EMAIL_ADDRESS', start: 0, end: 0, text: 'maria.gonzalez@example.com', probability: 0.99 },
]

test('a remote runtime\'s listing holds its profile, read field by field, and a local runtime has none', () => {
  const runtime = parseSettings({ runtime: 'remote', runtimeUrl: 'http://localhost:8899/v1/decisions', runtimeToken: '', runtimeModel: '' } as never).runtime
  expect(listingOf(runtime)).toBe('http://localhost:8899/v1/systemone/models')
  expect(listingOf({ ...runtime, kind: 'local-cpu' })).toBe(undefined)
  expect(profileOf({ vsr: { redact: false, router_turns: true, router_goal: 'cheap', guard: 'no', colour: true } })).toEqual({ redact: false, routerTurns: true })
  expect(profileOf({ models: [] })).toEqual({})
})

test('the listener turning redact off reaches the session, and says so once', { options: REMOTE }, async ($, on) => {
  const reply: Reply = { listing: { models: [], vsr: { redact: false } }, pii: SPANS, toasts: [] }
  engine(on, reply)
  files(on)
  await $.session.start(START)
  await $.session.start(START)

  expect(await readPii($)).toBe(PII)
  expect(reply.toasts).toEqual(['vSR follows the listener: redact off'])
})

test('without a profile the plugin keeps its own options', { options: REMOTE }, async ($, on) => {
  const reply: Reply = { pii: SPANS, toasts: [] }
  engine(on, reply)
  files(on)
  await $.session.start(START)

  expect(await readPii($)).toBe('Ticket from [vsr:name] <[vsr:email]>')
  expect(reply.toasts).toEqual([])
})

test('what the person changes with /vsr stays theirs, and /vsr follow hands it back', { options: REMOTE }, async ($, on) => {
  const reply: Reply = { listing: { models: [], vsr: { redact: false, guard: false } }, pii: SPANS, toasts: [] }
  engine(on, reply)
  files(on)
  const sets = configRows(on)
  await $.session.start(START)

  expect((await $.command.run({ command: 'vsr', args: 'redact on' })).text).toBe('vSR: redact on.')
  expect(sets).toEqual([{ key: 'vsr.redact', value: true }])
  // The option already held true, so nothing reloads: the change applies at once, and survives a reload.
  expect(await readPii($)).toBe('Ticket from [vsr:name] <[vsr:email]>')
  await $.session.start(START)
  expect(await readPii($)).toBe('Ticket from [vsr:name] <[vsr:email]>')

  expect((await $.command.run({ command: 'vsr', args: 'follow' })).text).toBe('vSR: follows the listener\'s profile: redact off, guard off.')
  expect(await readPii($)).toBe(PII)
})

test('a change in the /config menu is the person\'s too', { options: REMOTE }, async ($, on) => {
  const reply: Reply = { listing: { models: [], vsr: { redact: false } }, pii: SPANS }
  engine(on, reply)
  files(on)
  configRows(on)
  await $.session.start(START)
  await $.config.set({ key: 'vsr.redact', value: true, previous: false, provider: { kind: 'plugin', name: 'vsr' }, origin: { kind: 'composer' } } as never)
  expect(await readPii($)).toBe('Ticket from [vsr:name] <[vsr:email]>')
  await $.session.start(START)

  expect(await readPii($)).toBe('Ticket from [vsr:name] <[vsr:email]>')
})

test('a running session asks the listener again at a prompt a minute later', { options: REMOTE }, async ($, on) => {
  const reply: Reply = { listing: { models: [] }, pii: SPANS, toasts: [] }
  const clock = engine(on, reply)
  files(on)
  await $.session.start(START)
  reply.listing = { models: [], vsr: { redact: false } }
  const asked = reply.listings

  await $.prompt.submit({ text: 'hello', origin: { kind: 'composer' } } as never)
  expect(reply.listings).toBe(asked)
  await clock.advance(61_000)
  await $.prompt.submit({ text: 'hello again', origin: { kind: 'composer' } } as never)
  await clock.advance(0)

  expect(reply.listings).toBe((asked ?? 0) + 1)
  expect(await readPii($)).toBe(PII)
  expect(reply.toasts).toEqual(['vSR follows the listener: redact off'])
})
