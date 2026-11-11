import { expect, test } from 'claude-code/testing'
import type { On } from 'claude-code'

import { engine, paneText } from './kit'

const answered = (text: string) => ({ value: { status: 200, ok: true, headers: {}, text } })

// The kit's engine with every hook but its classifier, which answers here instead.
function engineWithout(on: On, fetch: Parameters<On>[1], reply: Parameters<typeof engine>[1]) {
  const rest = ((...args: Parameters<On>) => (args[0] === 'http.fetch' ? undefined : on(...args))) as On
  const clock = engine(rest, reply)
  on('http.fetch', fetch)
  return clock
}

test('a 200 answer whose body is not JSON is a failed question, not an offline runtime', { options: { guardWait: 'done' } }, async ($, on) => {
  const reply = { toasts: [] as string[] }
  engineWithout(on, (_, e) => answered(e.url.endsWith('/health') ? '{"status":"ok"}' : '<html><body>Sign in to the proxy</body></html>'), reply)

  const verdict = await $.tool.check({ tool: 'Bash', input: { command: 'ls' } })

  expect(verdict).toMatchObject({ decision: 'ask', reason: 'vSR guard: could not check 1 of 1 step: the decision model gave an answer vSR could not read. Confirm before it runs.' })
  expect(reply.toasts).toEqual([])
})

// A classifier whose answers wait until one of `asked` is called with a review score, in the order they were asked.
function heldEngine(on: On) {
  const asked: ((review: number) => void)[] = []
  const reply = (review: number) =>
    answered(JSON.stringify({ answers: { kind: { type: 'choice', choice: 'safe', probabilities: { safe: 0.9, destructive: 0.04, exfiltration: 0.03, remote_code: 0.03 }, confidence: 0.9 }, review: { type: 'noul', noul: review } } }))
  const clock = engineWithout(
    on,
    (_, e) => (e.url.endsWith('/health') ? answered('{"status":"ok"}') : new Promise(resolve => asked.push(review => resolve(reply(review))))),
    {},
  )
  return { asked, clock }
}

test('a question after one vSR gave up on waits for the runtime to finish that one, then is asked', async ($, on) => {
  const { asked, clock } = heldEngine(on)

  const first = $.tool.check({ tool: 'Bash', input: { command: 'ls' } })
  await clock.advance(3000)
  expect((await first).decision).toBe('allow')
  const second = $.tool.check({ tool: 'Bash', input: { command: 'pwd' } })
  await clock.advance(1000)
  const sentWhileBusy = asked.length
  asked[0](0.1)
  await clock.advance(0)
  await clock.advance(50)
  await clock.advance(50)
  const sentAfter = asked.length
  asked[1](0.1)
  await clock.advance(0)

  expect([sentWhileBusy, sentAfter]).toEqual([1, 2])
  expect(await second).toMatchObject({ decision: 'allow' })
})

test('a question the runtime never gets to, behind one vSR gave up on, is skipped at the end of its own wait', async ($, on) => {
  const { asked, clock } = heldEngine(on)

  const first = $.tool.check({ tool: 'Bash', input: { command: 'ls' } })
  await clock.advance(3000)
  await first
  const second = $.tool.check({ tool: 'Bash', input: { command: 'git push --force origin main' } })
  await clock.advance(3000)

  expect(await second).toMatchObject({ decision: 'allow' })
  expect(asked).toHaveLength(1)
  expect(await paneText($, /ran unchecked/)).toBe('command ran unchecked · the decision model was still answering an earlier question after 3 s · /vsr:setup checks it')
})

// A hosted System One API: no /health, a /v1/models listing, and answers only a question that names its model.
test(
  'a hosted runtime without /health counts as up through /v1/models, and its questions name the model',
  { options: { runtime: 'remote', runtimeUrl: 'https://sr.example/v1/decisions', runtimeModel: 'vllm-sr/vela3-flash', guardWait: 'done' } },
  async ($, on) => {
    const reply = { toasts: [] as string[] }
    const bodies: { model?: string }[] = []
    const safe = { answers: { kind: { type: 'choice', choice: 'safe', probabilities: { safe: 0.97, destructive: 0.01, exfiltration: 0.01, remote_code: 0.01 }, confidence: 0.97 }, review: { type: 'noul', noul: 0.01 } } }
    engineWithout(
      on,
      (_, e) => {
        if (e.url.endsWith('/health')) return { value: { status: 404, ok: false, headers: {}, text: '' } }
        if (e.url.endsWith('/v1/models')) return answered('{"data":[{"id":"vllm-sr/vela3-flash"}]}')
        bodies.push(JSON.parse(e.init?.body ?? '{}'))
        return answered(JSON.stringify(safe))
      },
      reply,
    )

    const verdict = await $.tool.check({ tool: 'Bash', input: { command: 'ls' } })

    expect(verdict).toMatchObject({ decision: 'allow' })
    expect(bodies.map(body => body.model)).toEqual(['vllm-sr/vela3-flash'])
    expect(reply.toasts).toEqual([])
  },
)
