import { expect, test } from 'claude-code/testing'

import {
  appToggles,
  bandRow,
  engine,
  HOME,
  LOG_FILE,
  logLines,
  paneLines,
  paneText,
  PUSH,
  type Reply,
  rowTree,
  SESSION,
  SPAWN,
  YES,
} from './kit'

type Test = Parameters<Parameters<typeof test>[1]>[0]

const AT = '1970-01-01T00:00:00.000Z'
const TICKET = 'From: Maria Gonzalez <maria.gonzalez@example.com>'
const TICKET_SPANS = [
  { label: 'PERSON', start: 6, end: 20, text: 'Maria Gonzalez', probability: 0.95 },
  { label: 'EMAIL_ADDRESS', start: 22, end: 48, text: 'maria.gonzalez@example.com', probability: 0.99 },
]
const readResult = (content: string) => ({
  result: { type: 'text', file: { filePath: '/repo/ticket.txt', content, numLines: 1, startLine: 1, totalLines: 1 } },
})
const WRONG = { SAT: 0.01, NEED_CLARIFICATION: 0.03, WRONG_ANSWER: 0.92, WANT_DIFFERENT: 0.03, NO_FEEDBACK: 0.01 }
const PUSHBACK = 'No, that summary is wrong.'
const ANSWER = 'Maria cannot log in after a password reset and asks for a callback.'

const submit = ($: Test, text: string) => $.prompt.submit({ text, origin: { kind: 'composer' } } as never)
const check = ($: Test, command: string) => $.tool.check({ tool: 'Bash', input: { command } })
const start = ($: Test) => $.session.start({ cwd: '/', surface: 'terminal', isInteractive: true })

test('every decision is one log line of the calibration shape, and a guard line gains its outcome once the call settles', async ($, on) => {
  const reply: Reply = { review: 0.98, files: {} }
  const clock = engine(on, reply)
  on('classic.PermissionRequest', () => YES)
  let settle = () => {}
  on('tool.call', () => new Promise<{ result: object }>(resolve => (settle = () => resolve({ result: { stdout: '' } }))))

  const pending = $.tool.call(PUSH)
  await clock.advance(1)
  const held = {
    id: 'call-9',
    at: AT,
    session: SESSION,
    app: 'guard',
    verdict: 'held',
    score: 0.98,
    threshold: 0.9,
    subject: 'git push --force origin main',
    detail: { risk: 'destructive', riskProbability: 0.84, review: 0.98, parts: 1, outcome: null },
    learnable: true,
  }
  expect(logLines(reply)).toEqual([held])

  settle()
  await pending
  await clock.advance(0)
  expect(logLines(reply)).toEqual([{ ...held, detail: { ...held.detail, outcome: 'approved' } }])
})

test('each app writes its own line: score, threshold, subject and detail', { options: { check: true } }, async ($, on) => {
  const reply: Reply = {
    complexity: { score: 0.09, confidence: 0.86 },
    steps: { score: 0.17, confidence: 0.74 },
    pii: TICKET_SPANS,
    jailbreak: 0.99,
    review: 0.2,
    feedback: WRONG,
    hallucination: 0.53,
    files: {},
  }
  const clock = engine(on, reply)
  on('agent.spawn', (_, e) => ({ model: e.model ?? e.parentModel, agentId: 'agent-1' }))
  on('turn.step', async function* (_, e) {
    return { turnId: e.turnId, index: e.index, answer: '', toolUses: [] }
  })
  on('tool.call', () => readResult(TICKET))

  await $.agent.spawn(SPAWN)
  for await (const _ of $.turn.step({ turnId: 's1', index: 0, model: 'haiku', effort: 'medium', messageCount: 1, agentId: 'agent-1' })) void _
  await submit($, PUSHBACK)
  await $.turn.start({ turnId: 't1', text: 'Summarize ticket.txt.' })
  await $.tool.call({ tool: 'Read', file_path: '/repo/ticket.txt', tool_use_id: 'r1' })
  await check($, 'ls')
  await $.turn.complete({ turnId: 't1', answer: ANSWER, reason: 'answer', durationMs: 1, isAborted: false })
  reply.fails = true
  await check($, 'git status')
  await clock.advance(0)

  const base = { at: AT, session: SESSION, learnable: true }
  expect(logLines(reply)).toEqual([
    {
      ...base,
      id: 'agent-1',
      app: 'router',
      verdict: 'routed',
      score: 0.09,
      threshold: 0.5,
      subject: 'Rename a variable',
      detail: {
        level: 0.09,
        confidence: 0.86,
        model: 'haiku',
        effort: 'medium',
        steps: { score: 0.17, confidence: 0.74 },
        effortFrom: 'haiku',
        escalated: null,
      },
    },
    {
      ...base,
      id: AT,
      app: 'escalate',
      verdict: 'escalated',
      score: 0.92,
      threshold: 0.8,
      subject: PUSHBACK,
      detail: { feedback: WRONG, reask: null },
    },
    {
      ...base,
      id: 'r1:redact',
      app: 'redact',
      verdict: 'hidden',
      score: 0.95,
      threshold: 0.7,
      subject: 'ticket.txt',
      detail: { counts: { name: 1, email: 1 }, source: 'ticket.txt' },
    },
    { ...base, id: 'r1:shield', app: 'shield', verdict: 'shielded', score: 0.99, threshold: 0.9, subject: 'ticket.txt', detail: { source: 'ticket.txt' } },
    {
      ...base,
      id: AT,
      app: 'guard',
      verdict: 'allowed',
      score: 0.2,
      threshold: 0.9,
      subject: 'ls',
      detail: { risk: 'destructive', riskProbability: 0.84, review: 0.2, parts: 1, outcome: null },
    },
    {
      ...base,
      id: 't1',
      app: 'check',
      verdict: 'unverified',
      score: 0.53,
      threshold: 0.5,
      subject: 'Summarize ticket.txt.',
      detail: { files: 1, commands: 0 },
    },
    {
      ...base,
      id: AT,
      app: 'guard',
      verdict: 'off',
      score: null,
      threshold: null,
      subject: 'git status',
      learnable: false,
      detail: { why: expect.stringMatching(/connection refused/) },
    },
  ])
  expect(reply.files?.[LOG_FILE]).not.toContain('Maria')
})

test('a log subject never carries a secret the rules hide', async ($, on) => {
  const reply: Reply = { review: 0.2, files: {} }
  const clock = engine(on, reply)

  await check($, 'DB_PASSWORD=hunter22 ./deploy.sh')
  await clock.advance(0)

  expect(logLines(reply).map(line => line.subject)).toEqual(['DB_PASSWORD=[secret] ./deploy.sh'])
})

test('a route line carries the steps answer and where the effort came from, a kept one carries neither', { options: { routerGoal: 'save' } }, async ($, on) => {
  const reply: Reply = { complexity: { score: 0.62, confidence: 0.39 }, steps: { score: 0.52, confidence: 0.23 }, files: {} }
  const clock = engine(on, reply)
  on('agent.spawn', (_, e) => ({ model: e.model ?? e.parentModel, agentId: `agent-${e.tool_use_id}` }))
  on('turn.step', async function* (_, e) {
    return { turnId: e.turnId, index: e.index, answer: '', toolUses: [] }
  })

  await $.agent.spawn({ ...SPAWN, tool_use_id: 'routed' })
  for await (const _ of $.turn.step({ turnId: 's1', index: 0, model: 'sonnet', effort: 'low', messageCount: 1, agentId: 'agent-routed' })) void _
  reply.complexity = { score: 1, confidence: 0.1 }
  await $.agent.spawn({ ...SPAWN, tool_use_id: 'kept' })
  await clock.advance(0)

  expect(logLines(reply).map(line => [line.verdict, line.detail])).toEqual([
    [
      'routed',
      {
        level: 0.62,
        confidence: 0.39,
        model: 'sonnet',
        effort: 'medium',
        steps: { score: 0.52, confidence: 0.23 },
        effortFrom: 'unsure',
        escalated: null,
      },
    ],
    ['kept', { level: 1, confidence: 0.1, model: 'opus', effort: null, steps: null, effortFrom: null, escalated: null, reason: 'unsure' }],
  ])
})

test('a ~/ log path lands under HOME; with HOME unset the file is not written, though the lines are kept', async ($, on) => {
  const reply: Reply = { review: 0.2, files: {}, env: {} }
  const clock = engine(on, reply)

  await check($, 'ls')
  await clock.advance(0)

  expect(reply.files).toEqual({})
  expect(await paneText($, /^calibrated/, 120)).toBe('calibrated  never · 1 decision since · log ~/.claude/vsr/decisions.jsonl')
})

test('with CLAUDE_CONFIG_DIR set, the default log goes into that config directory', async ($, on) => {
  const reply: Reply = { review: 0.2, files: {}, env: { HOME, CLAUDE_CONFIG_DIR: '/tmp/cfg' } }
  const clock = engine(on, reply)

  await check($, 'ls')
  await clock.advance(0)

  expect(Object.keys(reply.files ?? {})).toEqual(['/tmp/cfg/vsr/decisions.jsonl'])
})

test('a relative log path is refused, so no project gets another\'s commands', { options: { logPath: 'logs/vsr.jsonl' } }, async ($, on) => {
  const reply: Reply = { review: 0.2, files: {} }
  const clock = engine(on, reply)

  await check($, 'ls')
  await clock.advance(0)

  expect(reply.files).toEqual({})
})

test('a file of its own at the log path is left alone, not replaced by the log', async ($, on) => {
  const reply: Reply = { review: 0.2, files: { [LOG_FILE]: 'my own notes\n' } }
  const clock = engine(on, reply)
  on('fs.read', () => ({ value: 'my own notes\n' }))

  await check($, 'ls')
  await clock.advance(0)

  expect(reply.files).toEqual({ [LOG_FILE]: 'my own notes\n' })
})

test('each log line says whether calibration learns from it', async ($, on) => {
  const reply: Reply = { review: 0.2, files: {} }
  const clock = engine(on, reply)
  on('classic.PermissionRequest', () => ({ decision: 'deny' }) as never)

  await check($, 'ls')
  await $.tool.check({ tool: 'Bash', input: { command: 'rm -rf ~' }, tool_use_id: 'r1' } as never)
  await clock.advance(0)

  expect(logLines(reply).map(line => [line.subject, line.learnable])).toEqual([
    ['ls', true],
    ['rm -rf ~', false],
  ])
})

test('an empty log path writes no line and reminds of nothing', { options: { logPath: '', calibrateEvery: 1 } }, async ($, on) => {
  const reply: Reply = { review: 0.2, files: {}, toasts: [] }
  const clock = engine(on, reply)

  await check($, 'ls')
  await clock.advance(0)

  expect(reply.files).toEqual({})
  expect(reply.toasts).toEqual([])
  expect(await paneText($, /^calibrated/)).toBe('calibrated  log off')
})

test('the log keeps the newest 5000 lines across sessions', async ($, on) => {
  const old = Array.from({ length: 5000 }, (_, n) => ({
    id: `old-${n}`,
    at: AT,
    session: 'earlier',
    app: 'guard',
    verdict: 'allowed',
    score: 0.1,
    threshold: 0.9,
    subject: 'ls',
    detail: {},
  }))
  const reply: Reply = { review: 0.2, files: {}, store: { log: old } }
  const clock = engine(on, reply)

  await check($, 'git status')
  await clock.advance(0)

  const lines = logLines(reply)
  expect(lines).toHaveLength(5000)
  expect(lines[0].id).toBe('old-1')
  expect(lines.at(-1)).toMatchObject({ session: SESSION, subject: 'git status' })
})

test('the reminder toasts once, when the decisions since the last calibration reach calibrateEvery', { options: { calibrateEvery: 3 } }, async ($, on) => {
  const reply: Reply = { review: 0.2, files: {}, toasts: [], store: { calibratedAt: '1960-01-01T00:00:00.000Z' } }
  const clock = engine(on, reply)

  await check($, 'ls')
  await check($, 'pwd')
  await clock.advance(0)
  expect(reply.toasts).toEqual([])
  await check($, 'git status')
  await check($, 'git log')
  await clock.advance(0)

  expect(reply.toasts).toEqual(['vSR: 3 decisions since the last calibration · run /vsr:calibrate'])
})

test('/vsr calibrated starts the count again, and the pane shows when', async ($, on) => {
  const reply: Reply = { review: 0.2, files: {} }
  const clock = engine(on, reply)
  await check($, 'ls')
  await clock.advance(0)
  await clock.set(86_400_000)

  const ran = await $.command.run({ command: 'vsr', args: 'calibrated' })

  expect(ran.text).toBe('vSR: calibrated at 1970-01-02T00:00:00.000Z.')
  expect(await paneText($, /^calibrated/, 120)).toBe(
    'calibrated  1970-01-02T00:00:00.000Z · 0 decisions since · log ~/.claude/vsr/decisions.jsonl',
  )
})

test('a session start reminds once when the log already holds enough uncalibrated decisions', { options: { calibrateEvery: 2 } }, async ($, on) => {
  const line = { id: 'x', at: AT, session: 'earlier', app: 'guard', verdict: 'allowed', score: 0.1, threshold: 0.9, subject: 'ls', detail: {} }
  const reply: Reply = { toasts: [], store: { log: [line, { ...line, id: 'y' }] } }
  engine(on, reply)

  await start($)

  expect(reply.toasts).toEqual(['vSR: 2 decisions since the last calibration · run /vsr:calibrate'])
})

test('a guard holds a review of 0.85 when guardReview is 0.8, and the card shows that line', { options: { guardReview: 0.8 } }, async ($, on) => {
  engine(on, { review: 0.85 })

  const verdict = await check($, 'git push --force origin main')

  expect(verdict.decision).toBe('ask')
  expect(verdict.reason).toBe('vSR guard: "git push --force origin main" looks destructive (risk score 0.85 ≥ 0.80). Confirm before it runs.')
  expect(await paneText($, /waiting for you/)).toContain('risk score 0.85 ≥ 0.80')
})

test('a guard allows a review of 0.85 at the default line', async ($, on) => {
  engine(on, { review: 0.85 })

  const verdict = await check($, 'git push --force origin main')

  expect(verdict.decision).toBe('allow')
  expect(await paneText($, /ran normally/)).toBe('ran normally, not risky enough to hold · risk score 0.85 < 0.90')
})

test('the router, check and shield lines follow their options', {
  options: { routerEasyBelow: 0.05, routerHardFrom: 0.08, checkUnsupported: 0.6, shieldInject: 0.8, check: true },
}, async ($, on) => {
  const reply: Reply = { complexity: { score: 0.09, confidence: 0.86 }, hallucination: 0.53, jailbreak: 0.82 }
  engine(on, reply)
  on('agent.spawn', (_, e) => ({ model: e.model ?? e.parentModel, agentId: 'agent-1' }))
  on('tool.call', () => readResult('1. Write the tests.'))

  const started = await $.agent.spawn(SPAWN)
  await $.turn.start({ turnId: 't1', text: 'Summarize ticket.txt.' })
  const ran = await $.tool.call({ tool: 'Read', file_path: '/repo/ticket.txt', tool_use_id: 'r1' })
  await $.turn.complete({ turnId: 't1', answer: ANSWER, reason: 'answer', durationMs: 1, isAborted: false })

  expect(started.model).toBe('opus')
  expect(await paneText($, /subagent →/, 120)).toBe(
    'subagent → Opus · hard task · difficulty 0.09 of 2 · confidence 0.86 · reasoning steps unavailable → default high',
  )
  expect(ran.context).toHaveLength(1)
  expect(await paneText($, /answer matches/, 120)).toContain('unsupported 0.53 < 0.60')
})

test('an app turned off asks the classifier nothing and records nothing', {
  options: { guard: false, router: false },
}, async ($, on) => {
  const reply: Reply = { review: 0.98, complexity: { score: 0.09, confidence: 0.86 }, bodies: [], files: {} }
  const clock = engine(on, reply)
  on('agent.spawn', (_, e) => ({ model: e.model ?? e.parentModel, agentId: 'agent-1' }))

  const verdict = await check($, 'git push --force origin main')
  const started = await $.agent.spawn(SPAWN)
  await clock.advance(0)

  expect(verdict.decision).toBe('allow')
  expect(started.model).toBe('opus')
  expect(reply.bodies).toEqual([])
  expect(reply.files).toEqual({})
  expect(await paneText($, /No decisions/)).toBe('No decisions yet. Each one appears under the row it decided on.')
  expect(await appToggles($)).toEqual(['○ guard', '○ check', '● shield', '● redact', '○ router', '● escalate'])
})

test('a runtime token rides every request as a bearer header', { options: { runtimeToken: 'tok-123' } }, async ($, on) => {
  const reply: Reply = { review: 0.2, headers: [] }
  engine(on, reply)

  await check($, 'ls')

  expect(reply.headers).toEqual([{ 'content-type': 'application/json', authorization: 'Bearer tok-123' }])
})

test('with no token no authorization header is sent', async ($, on) => {
  const reply: Reply = { review: 0.2, headers: [] }
  engine(on, reply)

  await check($, 'ls')

  expect(reply.headers).toEqual([{ 'content-type': 'application/json' }])
})

test('requests go to the configured runtime URL, and its health check to the same base', {
  options: { runtimeUrl: 'https://sr.example.com/api/v1/decisions', runtime: 'remote' },
}, async ($, on) => {
  const reply: Reply = { review: 0.2, urls: [] }
  const clock = engine(on, reply)

  await start($)
  await clock.advance(0)
  await check($, 'ls')

  expect(reply.urls).toEqual(['https://sr.example.com/api/health', 'https://sr.example.com/api/v1/decisions'])
  expect(await paneText($, /^model/, 120)).toBe('model  remote · https://sr.example.com/api/v1/decisions')
})

test('a runtime that does not answer its health check toasts and draws the offline band', async ($, on) => {
  const reply: Reply = { offline: true, toasts: [] }
  const clock = engine(on, reply)

  await start($)
  await clock.advance(0)
  const { texts, props } = await rowTree($, bandRow())

  expect(reply.toasts).toEqual(["vSR: the decision model is not answering at http://127.0.0.1:8790/v1/decisions · vSR's checks step aside until it does (apps set to wait until done hold instead); guard's fixed rules still hold · /vsr:setup"])
  expect(texts[0]).toBe(' vSR   ○ model offline · guard, redact, shield not checking')
  expect(props[texts.indexOf('○')]).toMatchObject({ color: 'warning' })
  expect((await paneLines($)).filter(line => line.startsWith('model'))[0]).toContain('not answering')
})

test('a runtime that answers its health check leaves the band and the toasts alone', async ($, on) => {
  const reply: Reply = { toasts: [] }
  const clock = engine(on, reply)

  await start($)
  await clock.advance(0)

  expect(reply.toasts).toEqual([])
  expect((await rowTree($, bandRow())).texts).toEqual(['the engine row'])
})

test('an answer from the classifier, a minute after it went offline, clears the offline band', async ($, on) => {
  const reply: Reply = { offline: true, review: 0.2 }
  const clock = engine(on, reply)
  await start($)
  await clock.advance(60_000)

  await check($, 'ls')

  expect((await rowTree($, bandRow())).texts[0]).toBe(' vSR   › 1 allowed')
})
