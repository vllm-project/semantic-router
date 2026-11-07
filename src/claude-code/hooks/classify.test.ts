import { expect, test } from 'claude-code/testing'
import type { On } from 'claude-code'

import { engine } from './kit'

const answered = (text: string) => ({ value: { status: 200, ok: true, headers: {}, text } })

// The kit's engine with every hook but its classifier, which answers here instead.
function engineWithout(on: On, fetch: Parameters<On>[1], reply: Parameters<typeof engine>[1]) {
  const rest = ((...args: Parameters<On>) => (args[0] === 'http.fetch' ? undefined : on(...args))) as On
  engine(rest, reply)
  on('http.fetch', fetch)
}

test('a 200 answer whose body is not JSON is a failed question, not an offline runtime', async ($, on) => {
  const reply = { toasts: [] as string[] }
  engineWithout(on, (_, e) => answered(e.url.endsWith('/health') ? '{"status":"ok"}' : '<html><body>Sign in to the proxy</body></html>'), reply)

  const verdict = await $.tool.check({ tool: 'Bash', input: { command: 'ls' } })

  expect(verdict).toMatchObject({ decision: 'ask', reason: 'vSR guard: could not check 1 of 1 step (unexpected answer shape). Confirm before it runs.' })
  expect(reply.toasts).toEqual([])
})
