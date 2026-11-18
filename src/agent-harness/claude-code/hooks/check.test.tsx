import { expect, test } from 'claude-code/testing'

import { assistantRow, ENGINE_ROW, engine, paneText, type Reply, rowTree, type Span } from './kit'

type Test = Parameters<Parameters<typeof test>[1]>[0]

const PROMPT = 'Summarize support/ticket-1042.txt in two sentences.'
const TICKET = 'Ticket 1042: Maria Gonzalez cannot log in after a password reset. She asks for a callback.'
const FAITHFUL = 'The customer cannot log in after a password reset.\n\nThey ask for a **callback** and nothing else.'
const INVENTED = 'The account was locked after five failed attempts.\n\nA refund of $40 was issued to the customer.'

const readResult = (content: string) => ({
  result: { type: 'text', file: { filePath: '/repo/support/ticket-1042.txt', content, numLines: 1, startLine: 1, totalLines: 1 } },
})

const NAME: Span = { label: 'PERSON', start: 13, end: 27, text: 'Maria Gonzalez', probability: 0.95 }

async function turnThatReads($: Test, answer: string, reads = 1) {
  await $.turn.start({ turnId: 't1', text: PROMPT })
  for (let n = 0; n < reads; n++) await $.tool.call({ tool: 'Read', file_path: '/repo/support/ticket-1042.txt', tool_use_id: `r${n}` })
  await $.turn.complete({ turnId: 't1', answer, reason: 'answer', durationMs: 1, isAborted: false })
}

test('an answer that keeps to what the turn read is GROUNDED, judged against the redacted reads', { options: { check: true } }, async ($, on) => {
  const reply: Reply = { hallucination: 0.05, pii: [NAME], bodies: [] }
  engine(on, reply)
  on('tool.call', () => readResult(TICKET))

  await turnThatReads($, FAITHFUL)

  expect(reply.bodies?.find(body => body.questions.hallucination)?.state).toEqual({
    context: 'support/ticket-1042.txt:\nTicket 1042: [vsr:name] cannot log in after a password reset. She asks for a callback.',
    request: PROMPT,
    answer: FAITHFUL,
  })
  expect(await paneText($, /GROUNDED/, 100)).toBe(` GROUNDED  ${PROMPT}`)
  expect(await paneText($, /answer matches/, 120)).toBe(
    'answer matches what Claude read this turn · unsupported 0.05 < 0.50 · checked against 1 file read this turn',
  )
})

test('check reads a command as it ran: after a cd a path shortened for the card would name another file', { options: { check: true } }, async ($, on) => {
  const reply: Reply = { hallucination: 0.05, bodies: [] }
  engine(on, reply)
  on('tool.call', () => ({ result: { stdout: TICKET, stderr: '', interrupted: false } }))
  const command = 'cd /other && cat /repo/support/ticket-1042.txt'

  await $.turn.start({ turnId: 't1', text: PROMPT })
  await $.tool.call({ tool: 'Bash', command, tool_use_id: 'b1' })
  await $.turn.complete({ turnId: 't1', answer: FAITHFUL, reason: 'answer', durationMs: 1, isAborted: false })

  expect(reply.bodies?.find(body => body.questions.hallucination)?.state.context).toBe(`${command}:\n${TICKET}`)
})

test('an answer with claims the reads do not back is UNVERIFIED', { options: { check: true } }, async ($, on) => {
  engine(on, { hallucination: 0.53 })
  on('tool.call', () => readResult(TICKET))

  await turnThatReads($, INVENTED, 2)

  expect(await paneText($, /UNVERIFIED/, 100)).toBe(` UNVERIFIED  ${PROMPT}`)
  expect(await paneText($, /not backed/, 120)).toContain('unsupported 0.53 ≥ 0.50 · checked against 2 files read this turn')
})

test('the check card is drawn under the last block of the answer it judged, and no other block', { options: { check: true } }, async ($, on) => {
  engine(on, { hallucination: 0.53 })
  on('tool.call', () => readResult(TICKET))
  await turnThatReads($, INVENTED)

  const last = await rowTree($, assistantRow('A refund of $40 was issued to the customer.'))
  const first = await rowTree($, assistantRow('The account was locked after five failed attempts.', true))

  expect(last.texts[0]).toBe(ENGINE_ROW)
  expect(last.cards).toEqual([expect.objectContaining({ borderStyle: 'round', borderColor: 'warning', width: 76 })])
  expect(last.texts).toContain('!  UNVERIFIED  some claims are not backed by what Claude read')
  expect(last.texts).toContain('unsupported ▕████████████┃▊░░░░░░░░░░░▏  0.53  flags at 0.50')
  expect(last.texts).toContain('look twice before you rely on it')
  expect(first.cards).toEqual([])
})

test('a grounded answer draws a GROUNDED card with its meter in the success colour', { options: { display: 'verbose', check: true } }, async ($, on) => {
  engine(on, { hallucination: 0.05 })
  on('tool.call', () => readResult(TICKET))
  await turnThatReads($, FAITHFUL)

  const { texts, props, cards } = await rowTree($, assistantRow('They ask for a callback and nothing else.'))

  expect(cards).toEqual([expect.objectContaining({ borderColor: 'success' })])
  expect(texts).toContain('✓  GROUNDED  answer matches what Claude read this turn')
  expect(texts).toContain('checked against 1 file read this turn')
  expect(props[texts.indexOf('█▎')]).toMatchObject({ color: 'success' })
})

test('a turn that read nothing, or answered in a few words, is not checked', async ($, on) => {
  const reply: Reply = { bodies: [] }
  engine(on, reply)
  on('tool.call', () => readResult(TICKET))

  await turnThatReads($, FAITHFUL, 0)
  await turnThatReads($, 'Done.')

  expect(reply.bodies?.filter(body => body.questions.hallucination)).toEqual([])
})

test('check fails open: with no answer from the classifier the turn ends as it would and shows SKIPPED', { options: { display: 'verbose', check: true } }, async ($, on) => {
  const reply: Reply = {}
  engine(on, reply)
  on('tool.call', () => readResult(TICKET))

  reply.errors = true
  await turnThatReads($, INVENTED)
  const { texts } = await rowTree($, assistantRow('A refund of $40 was issued to the customer.'))

  expect(texts).toContain('○  SKIPPED  answer not checked')
  expect(await paneText($, /GROUNDED|UNVERIFIED/)).toBeUndefined()
})

test('check is off unless turned on: a turn that read files asks nothing about its answer', async ($, on) => {
  const reply: Reply = { hallucination: 0.9, bodies: [] }
  engine(on, reply)
  on('tool.call', () => readResult('1. Write the tests.'))

  await $.turn.start({ turnId: 't1', text: 'Summarize ticket.txt.' })
  await $.tool.call({ tool: 'Read', file_path: '/repo/ticket.txt', tool_use_id: 'r1' })
  await $.turn.complete({ turnId: 't1', answer: 'The ticket asks for tests first, then the fix, then a release note.', reason: 'answer', durationMs: 1, isAborted: false })

  expect(reply.bodies?.filter(body => JSON.stringify(body).includes('hallucination'))).toEqual([])
})

test('by default a grounded answer draws nothing under it, and an unbacked one draws its card', { options: { check: true } }, async ($, on) => {
  const reply: Reply = { hallucination: 0.05 }
  engine(on, reply)
  on('tool.call', () => readResult(TICKET))

  await turnThatReads($, FAITHFUL)
  const grounded = await rowTree($, assistantRow('They ask for a **callback** and nothing else.'))
  reply.hallucination = 0.53
  await turnThatReads($, INVENTED)
  const unbacked = await rowTree($, assistantRow('A refund of $40 was issued to the customer.'))

  expect(grounded.texts).toEqual([ENGINE_ROW])
  expect(unbacked.cards).toEqual([expect.objectContaining({ borderColor: 'warning' })])
})
