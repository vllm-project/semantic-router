import { expect, test } from 'claude-code/testing'
import type { Hook, On } from 'claude-code'

import { engine, HOME, PUSH, type Reply, SESSION, type Test, YES } from './kit'
import { GATHER_MS, TOOL_EVENTS_PATH } from './timing'

const MANAGEMENT = 'http://127.0.0.1:8080'
const OPTIONS = { options: { managementUrl: `${MANAGEMENT}/` } }
const iso = (ms: number) => new Date(ms).toISOString()

type Posted = { body: Record<string, unknown>; headers?: Record<string, string> }

// The kit's engine, with its `http.fetch` behind one that takes the management API's requests: the tool-events
// batches the plugin posted, parsed, with their headers, answered (or never, for a Router that hangs).
function engineCapturing(on: On, reply: Reply, posted: Posted[], hang = false) {
  let kitFetch: Hook<'http.fetch'> | undefined
  const rest = ((...args: Parameters<On>) => {
    if (args[0] !== 'http.fetch') return on(...args)
    kitFetch = args[1] as Hook<'http.fetch'>
  }) as On
  const clock = engine(rest, reply)
  on('http.fetch', ($, e, next) => {
    if (!e.url.startsWith(`${MANAGEMENT}/`)) return (kitFetch as Hook<'http.fetch'>)($, e, next)
    expect(e.url).toBe(`${MANAGEMENT}${TOOL_EVENTS_PATH}`)
    posted.push({ body: JSON.parse(e.init?.body ?? '{}'), headers: e.init?.headers })
    return hang ? new Promise(() => {}) : { value: { status: 200, ok: true, headers: {}, text: '{"accepted":1}' } }
  })
  return clock
}

// Core's own PostToolUse as the call ends: Claude Code's measure of how long the tool ran.
const ranFor = ($: Test, id: string, tool: string, ms: number) =>
  $.classic.PostToolUse({
    hook_event_name: 'PostToolUse', session_id: SESSION, transcript_path: '/t', cwd: '/repo',
    tool_name: tool, tool_input: { secret: 'never sent' }, tool_response: 'never sent', tool_use_id: id, duration_ms: ms,
  } as never)

test('a held command sends its approval wait apart from how long it ran, and nothing of its input or output', OPTIONS, async ($, on) => {
  const posted: Posted[] = []
  const clock = engineCapturing(on, { review: 0.98, env: { HOME, VSR_MANAGEMENT_TOKEN: 'mgmt-key' } }, posted)
  on('classic.PostToolUse', () => ({}))
  let approve = () => {}
  on('classic.PermissionRequest', () => new Promise(resolve => (approve = () => resolve(YES))))
  let finish = async () => {}
  on('tool.call', () => new Promise(resolve => (finish = async () => resolve((await ranFor($, PUSH.tool_use_id, 'Bash', 400), { result: { stdout: 'pushed' } })))))
  const start = clock.now()

  const pending = $.tool.call(PUSH)
  await clock.advance(5000)
  approve()
  await clock.advance(400)
  await finish()
  await pending
  await clock.advance(GATHER_MS)

  expect(posted).toHaveLength(1)
  expect(posted[0].headers).toMatchObject({ authorization: 'Bearer mgmt-key', 'content-type': 'application/json' })
  expect(posted[0].body).toEqual({
    harness: 'claude-code',
    session_id: SESSION,
    events: [
      { tool_call_id: 'call-9', tool: 'Bash', status: 'succeeded', approval_requested_at: iso(start), started_at: iso(start + 5000), ended_at: iso(start + 5400) },
    ],
  })
  expect(JSON.stringify(posted[0].body)).not.toMatch(/git push|never sent|pushed/)
})

test('tools that run side by side each send their own times, in one batch', OPTIONS, async ($, on) => {
  const posted: Posted[] = []
  const clock = engineCapturing(on, {}, posted)
  on('classic.PostToolUse', () => ({}))
  const start = clock.now()
  const finishes = new Map<string, () => Promise<void>>()
  on('tool.call', (_, e) => new Promise(resolve => finishes.set(String(e.tool_use_id), async () => resolve((await ranFor($, String(e.tool_use_id), 'Read', clock.now() - start), { result: {} })))))

  const reads = [$.tool.call({ tool: 'Read', file_path: '/repo/a.py', tool_use_id: 'r1' } as never), $.tool.call({ tool: 'Read', file_path: '/repo/b.py', tool_use_id: 'r2' } as never)]
  await clock.advance(100)
  await finishes.get('r1')?.()
  await clock.advance(200)
  await finishes.get('r2')?.()
  await Promise.all(reads)
  await clock.advance(GATHER_MS)

  expect(posted.map(one => one.body.events)).toEqual([
    [
      { tool_call_id: 'r1', tool: 'Read', status: 'succeeded', started_at: iso(start), ended_at: iso(start + 100) },
      { tool_call_id: 'r2', tool: 'Read', status: 'succeeded', started_at: iso(start), ended_at: iso(start + 300) },
    ],
  ])
})

test('a refused call is sent as denied, with no run, under the session vllm-sr connect pinned', OPTIONS, async ($, on) => {
  const posted: Posted[] = []
  const clock = engineCapturing(on, { env: { HOME, VSR_SESSION_ID: 'pinned-session' } }, posted)
  on('tool.call', () => ({ deny: 'a rule refuses it' }))
  const now = clock.now()

  await $.tool.call({ tool: 'Write', file_path: '/etc/hosts', content: 'x', tool_use_id: 'w1' } as never)
  await clock.advance(GATHER_MS)

  expect(posted.map(one => one.body)).toEqual([
    { harness: 'claude-code', session_id: 'pinned-session', events: [{ tool_call_id: 'w1', tool: 'Write', status: 'denied', ended_at: iso(now) }] },
  ])
})

test('a tool that reports its failure is sent as failed, with how long it ran', OPTIONS, async ($, on) => {
  const posted: Posted[] = []
  const clock = engineCapturing(on, {}, posted)
  on('classic.PostToolUseFailure', () => ({}))
  on('tool.call', async () => {
    await $.classic.PostToolUseFailure({ hook_event_name: 'PostToolUseFailure', session_id: SESSION, transcript_path: '/t', cwd: '/repo', tool_name: 'Bash', tool_input: {}, tool_use_id: 'b1', error: 'Exit code 2', duration_ms: 250 } as never)
    return { isError: true, result: 'Exit code 2', text: 'Exit code 2' }
  })
  const now = clock.now()

  await $.tool.call({ tool: 'Bash', command: 'make test', tool_use_id: 'b1' } as never)
  await clock.advance(GATHER_MS)

  expect(posted[0]?.body.events).toEqual([{ tool_call_id: 'b1', tool: 'Bash', status: 'failed', started_at: iso(now - 250), ended_at: iso(now) }])
})

test('a Router that does not answer holds no call up', OPTIONS, async ($, on) => {
  const posted: Posted[] = []
  const clock = engineCapturing(on, {}, posted, true)
  on('tool.call', () => ({ result: {} }))

  for (const id of ['g1', 'g2', 'g3']) {
    expect(await $.tool.call({ tool: 'Glob', pattern: '*.py', tool_use_id: id } as never)).toEqual({ result: {} })
    await clock.advance(GATHER_MS)
  }

  // One batch is out and unanswered; the calls after it wait for it, at most two seconds.
  expect(posted.map(one => (one.body.events as { tool_call_id: string }[]).map(event => event.tool_call_id))).toEqual([['g1'], ['g2', 'g3']])
})

test('the end of a turn sends what waits at once', OPTIONS, async ($, on) => {
  const posted: Posted[] = []
  engineCapturing(on, {}, posted)
  on('tool.call', () => ({ result: {} }))

  await $.tool.call({ tool: 'Glob', pattern: '*.py', tool_use_id: 'g1' } as never)
  expect(posted).toEqual([])
  await $.turn.complete({ turnId: 't1', answer: 'done', reason: 'answer', durationMs: 1, isAborted: false })

  expect(posted.map(one => one.body.events)).toEqual([[expect.objectContaining({ tool_call_id: 'g1', tool: 'Glob' })]])
})

test('with no management API the plugin sends no timing', { options: {} }, async ($, on) => {
  const posted: Posted[] = []
  const clock = engineCapturing(on, {}, posted)
  on('tool.call', () => ({ result: {} }))

  await $.tool.call({ tool: 'Glob', pattern: '*.py', tool_use_id: 'g1' } as never)
  await clock.advance(GATHER_MS)

  expect(posted).toEqual([])
})
