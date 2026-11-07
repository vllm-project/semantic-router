import { expect, test } from 'claude-code/testing'
import type { On } from 'claude-code'

import { engine, paneText } from './kit'

const answered = (text: string) => ({ value: { status: 200, ok: true, headers: {}, text } })
const GUARD_ANSWER = JSON.stringify({
  answers: {
    kind: { type: 'choice', choice: 'safe', probabilities: { safe: 0.9, destructive: 0.05, exfiltration: 0.03, remote_code: 0.02 }, confidence: 0.9 },
    review: { type: 'noul', noul: 0.1 },
  },
})

// The kit's engine with every hook but its classifier, which answers here instead.
function engineWithout(on: On, fetch: Parameters<On>[1], reply: Parameters<typeof engine>[1]) {
  const rest = ((...args: Parameters<On>) => (args[0] === 'http.fetch' ? undefined : on(...args))) as On
  engine(rest, reply)
  on('http.fetch', fetch)
}

test('a request cancelled with its call (the user pressed Esc) leaves the runtime online and records nothing', async ($, on) => {
  const reply = { toasts: [] as string[] }
  let calls = 0
  engineWithout(
    on,
    (_, e) => {
      if (e.url.endsWith('/health')) return answered('{"status":"ready"}')
      calls += 1
      // As the engine fails a fetch whose call was cancelled: the message seen live when Esc interrupted a command.
      if (calls === 1) return { deny: 'The operation was aborted.' }
      return answered(GUARD_ANSWER)
    },
    reply,
  )

  const cancelled = await $.tool.check({ tool: 'Bash', input: { command: 'git status' } })
  const next = await $.tool.check({ tool: 'Bash', input: { command: 'ls' } })

  expect(cancelled).not.toMatchObject({ decision: 'ask' })
  expect(next).not.toMatchObject({ decision: 'ask' })
  expect(reply.toasts).toEqual([])
  expect(await paneText($, /SKIPPED|HELD/)).toBeUndefined()
})

test('a refused command\'s result is not read for personal data: nothing of the tool\'s went to Claude', async ($, on) => {
  const reply = { toasts: [] as string[], bodies: [] as unknown[] }
  engine(on, { ...reply, review: 0.1, pii: 'fails' })
  on('tool.call', () => ({ isError: true, result: 'The user doesn\'t want to proceed with this tool use.', text: 'The user doesn\'t want to proceed with this tool use.' }))

  await $.tool.call({ tool: 'Bash', command: 'git reset --hard HEAD~1', tool_use_id: 'b1' } as never)

  expect(await paneText($, /SKIPPED|REDACTED/)).toBeUndefined()
})

test('a redact request cancelled with its command records nothing: the run was cut short, not left unchecked', async ($, on) => {
  const reply = { toasts: [] as string[] }
  engineWithout(
    on,
    (_, e) => {
      if (e.url.endsWith('/health')) return answered('{"status":"ready"}')
      const body = JSON.parse(e.init?.body ?? '{}') as { questions?: Record<string, unknown> }
      if (body.questions?.pii) return { deny: 'The operation was aborted.' }
      return answered(GUARD_ANSWER)
    },
    reply,
  )
  on('tool.call', () => ({ isError: true, result: 'Exit code 130\nwarning: name Maria Gonzalez', text: 'Exit code 130\nwarning: name Maria Gonzalez' }))

  await $.tool.call({ tool: 'Bash', command: 'make long-test', tool_use_id: 'b2' } as never)

  expect(await paneText($, /SKIPPED/)).toBeUndefined()
  expect(reply.toasts).toEqual([])
})
