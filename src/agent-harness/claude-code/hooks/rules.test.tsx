import { expect, test } from 'claude-code/testing'
import type { On } from 'claude-code'

import { BOTTOM, engine, paneText, readCallOf, type Reply, rowTree, type Test, toolRow } from './kit'
import { rulesOf } from './lib/rules'

const REMOTE = { runtime: 'remote', runtimeUrl: 'http://localhost:8899/v1/decisions', runtimeModel: 'vela3-lite' } as const
const START = { cwd: '/', surface: 'terminal', isInteractive: true } as const
const PSQL = 'psql --host=prod-db.internal -c "select 1"'
const PROD = { name: 'prod_database', description: 'touches the production database.', required: true }
const EMPLOYEE = { name: 'employee_id', description: 'an employee ID', required: false }

// A listing that binds the given rules under revision r1.
const listingOf = (guard: object[], redact: object[] = [], revision = 'r1') => ({ models: [], vsr: { rules: { revision, guard, redact } } })

// The rules endpoint's answer: for each part, the guard rules `matched` names, or what `spans` finds, and the rules
// `unchecked` names.
const answerOf = (rules: { matched?: string[]; spans?: (text: string) => object[]; unchecked?: string[]; revision?: string }) =>
  (request: { parts: { id: string; text: string }[] }) => ({
    revision: rules.revision ?? 'r1',
    parts: request.parts.map(({ id, text }) => {
      const unchecked = (rules.unchecked ?? []).map(rule => ({ rule, reason: 'the decision model is unavailable' }))
      return { id, coverage: unchecked.length > 0 ? 'partial' : 'complete', matches: rules.matched ?? [], spans: rules.spans?.(text) ?? [], unchecked }
    }),
  })

const readOf = (content: string) => ({ result: { type: 'text', file: { filePath: '/repo/t.txt', content, numLines: 1, startLine: 1, totalLines: 1 } } })
const fileHolds = (on: On, content: string) => on('tool.call', { tool: 'Read' }, () => readOf(content) as never)
const read = async ($: Test) => $.tool.call({ tool: 'Read', tool_use_id: 'r1', input: { file_path: '/repo/t.txt' } } as never)

test('a listing\'s rules are read whole or not at all: a malformed entry keeps the rules read before', () => {
  expect(rulesOf({ vsr: { redact: true } })).toEqual({ revision: '', guard: [], redact: [] })
  expect(rulesOf(listingOf([PROD]))).toEqual({ revision: 'r1', guard: [PROD], redact: [] })
  expect(rulesOf(listingOf([PROD, { name: 'broken', required: 'yes' }]))).toBe(undefined)
  expect(rulesOf({ vsr: { rules: { guard: [PROD], redact: [] } } })).toBe(undefined)
})

test('a command a listener rule matches is held with the rule\'s description, whatever the model scores', { options: REMOTE }, async ($, on) => {
  const reply: Reply = { listing: listingOf([PROD]), review: 0.1, rules: { answer: answerOf({ matched: ['prod_database'] }), asked: [] } }
  engine(on, reply)
  await $.session.start(START)

  const verdict = await $.tool.check({ tool: 'Bash', input: { command: PSQL } })

  expect(verdict).toMatchObject({ decision: 'ask' })
  expect(verdict.reason).toBe(`vSR guard: "${PSQL}" touches the production database: a rule of the Router listener, always held. Confirm before it runs.`)
  // The whole command is one part: a rule reads it as it will run.
  expect(reply.rules?.asked).toEqual([{ app: 'guard', parts: [{ id: 'p0', text: PSQL }] }])
})

test('a read-only command the model skips is still read by the listener\'s rules', { options: REMOTE }, async ($, on) => {
  const reply: Reply = { listing: listingOf([PROD]), rules: { answer: answerOf({ matched: ['prod_database'] }) } }
  engine(on, reply)
  await $.session.start(START)

  const verdict = await $.tool.check({ tool: 'Bash', input: { command: 'cat /etc/prod-db.internal.conf' }, isReadOnly: true } as never)

  expect(verdict).toMatchObject({ decision: 'ask' })
})

test('a required rule the Router could not check holds the command; an advisory one lets it run', { options: REMOTE }, async ($, on) => {
  const reply: Reply = { listing: listingOf([PROD]), review: 0.1, rules: { answer: 'fails' } }
  engine(on, reply)
  await $.session.start(START)
  const held = await $.tool.check({ tool: 'Bash', input: { command: PSQL } })
  expect(held).toMatchObject({ decision: 'ask' })
  expect(held.reason).toContain("could not check the listener's required rule prod_database")
  expect(await paneText($, /required rule/)).toContain('required rule prod_database not checked')

  reply.listing = listingOf([{ ...PROD, required: false }])
  await $.session.start(START)
  expect(await $.tool.check({ tool: 'Bash', input: { command: PSQL } })).toEqual(BOTTOM)
})

test('an answer for another revision of the rules holds a command a required rule binds', { options: REMOTE }, async ($, on) => {
  const reply: Reply = { listing: listingOf([PROD]), review: 0.1, rules: { answer: answerOf({ revision: 'r2' }) } }
  engine(on, reply)
  await $.session.start(START)

  const verdict = await $.tool.check({ tool: 'Bash', input: { command: 'ls' } })

  expect(verdict).toMatchObject({ decision: 'ask' })
  expect(verdict.reason).toContain("the listener's rules changed")
})

test('the rules read before still hold when the listing does not answer', { options: REMOTE }, async ($, on) => {
  const reply: Reply = { listing: listingOf([PROD]), review: 0.1, rules: { answer: answerOf({ matched: ['prod_database'] }) } }
  engine(on, reply)
  await $.session.start(START)
  reply.listing = undefined
  reply.fails = true
  await $.session.start(START)
  reply.fails = false

  expect(await $.tool.check({ tool: 'Bash', input: { command: PSQL } })).toMatchObject({ decision: 'ask' })
})

test('what a redact rule finds is hidden as its tag, past an astral character, and overlaps join', { options: REMOTE }, async ($, on) => {
  const text = 'Paid 😀 by E-104233 today'
  // Code points: E-104233 starts at 10 (the emoji is one code point, two UTF-16 units).
  const reply: Reply = {
    listing: listingOf([], [EMPLOYEE, { ...EMPLOYEE, name: 'digits' }]),
    rules: { answer: answerOf({ spans: () => [{ start: 10, end: 18, rule: 'employee_id' }, { start: 12, end: 19, rule: 'digits' }] }) },
  }
  engine(on, reply)
  fileHolds(on, text)
  await $.session.start(START)

  const ran = await read($)

  expect((ran.result as { file: { content: string } }).file.content).toBe('Paid 😀 by [vsr:employee_id]today')
  // A tag longer than the value it hides is not text the checks left unread.
  const { texts } = await rowTree($, toolRow(readCallOf('r1', '/repo/t.txt')))
  expect(texts.some(one => /went to Claude unchecked/.test(one))).toBe(false)
})

test('an output a required redact rule could not check is withheld, with no advice to wait less', { options: REMOTE }, async ($, on) => {
  const reply: Reply = { listing: listingOf([], [{ ...EMPLOYEE, required: true }]), rules: { answer: 'fails' } }
  engine(on, reply)
  fileHolds(on, 'Paid by E-104233')
  await $.session.start(START)

  const ran = await read($)

  expect(ran.deny).toContain("against the listener's required rule employee_id")
  expect(ran.deny).not.toContain('shorter wait')
})

test('the personal data check reads what the rules read, and a rule and a PII span that cross are hidden as one', { options: REMOTE }, async ($, on) => {
  const text = 'Mail E-104233@corp.example now'
  const reply: Reply = {
    listing: listingOf([], [EMPLOYEE]),
    rules: { answer: answerOf({ spans: () => [{ start: 5, end: 13, rule: 'employee_id' }] }) },
    pii: [{ label: 'EMAIL_ADDRESS', start: 0, end: 0, text: 'E-104233@corp.example', probability: 0.99 }],
    bodies: [],
  }
  engine(on, reply)
  fileHolds(on, text)
  await $.session.start(START)

  const ran = await read($)

  expect((ran.result as { file: { content: string } }).file.content).toBe('Mail [vsr:email] now')
  expect(reply.bodies?.some(body => String(body.state).includes('E-104233@corp.example'))).toBe(true)
})

test('a listing that is not one keeps the rules read before', () => {
  expect(rulesOf(null)).toBe(undefined)
  expect(rulesOf({ vsr: 'broken' })).toBe(undefined)
  expect(rulesOf({ models: [] })).toEqual({ revision: '', guard: [], redact: [] })
})
