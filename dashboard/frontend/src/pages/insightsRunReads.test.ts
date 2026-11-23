import { describe, expect, it } from 'vitest'

import { InsightsRequestError } from './insightsPageApi'
import {
  beginRunRead,
  initialRunReads,
  settleRunRead,
  shownRunReads,
  type RunRead,
  type RunReadState,
} from './insightsRunReads'
import type { RunResponse } from './insightsRunTypes'

const run = (sessionId: string): RunResponse => ({
  object: 'router_replay.run',
  run: {
    session_id: sessionId,
    started_at: '2026-11-19T02:36:38Z',
    ended_at: '2026-11-19T02:36:42Z',
    duration_ms: 4_000,
    calls: 1,
    agents: [],
    models: [],
    outcome: { status: 'answered', tool_results: 0, tool_failures: 0 },
    cost: { unpriced_records: 0, unpriced_router_calls: 0, prompt_tokens: 0, completion_tokens: 0 },
  },
  calls: [],
})

const read = (sessionId: string, number: number): RunRead => ({ sessionId, number })
const notFound = () => new InsightsRequestError('agent run', 404, 'Not Found')
const shownSession = (state: RunReadState, sessionId: string) =>
  shownRunReads(state, sessionId).response?.run.session_id

/** The repro: session A's read is pending, the page switches to B, and B answers first. */
function switchedWhilePending() {
  const a = read('session-a', 1)
  const b = read('session-b', 2)
  let state = initialRunReads('session-a')
  state = beginRunRead(state, a)
  state = beginRunRead(state, b)
  state = settleRunRead(state, b, { response: run('session-b') })
  return { state, a, b }
}

describe("the run page's reads", () => {
  it('keeps the active run when the previous session answers late', () => {
    const { state, a } = switchedWhilePending()

    const late = settleRunRead(state, a, { response: run('session-a') })

    expect(late).toBe(state)
    expect(shownSession(late, 'session-b')).toBe('session-b')
  })

  it("does not let the previous session's 404 clear the active run", () => {
    const { state, a } = switchedWhilePending()

    const late = settleRunRead(state, a, { error: notFound() })

    expect(late).toBe(state)
    expect(shownRunReads(late, 'session-b').notFound).toBe(false)
    expect(shownSession(late, 'session-b')).toBe('session-b')
  })

  it('shows the session that answers first, then the one the page switched to', () => {
    const a = read('session-a', 1)
    const b = read('session-b', 2)
    let state = initialRunReads('session-a')
    state = beginRunRead(state, a)
    state = settleRunRead(state, a, { response: run('session-a') })
    state = beginRunRead(state, b)
    state = settleRunRead(state, b, { response: run('session-b') })

    expect(shownSession(state, 'session-b')).toBe('session-b')
  })

  it('still shows not-found for the 404 of the session the page names', () => {
    const b = read('session-b', 2)
    let state = settleRunRead(beginRunRead(initialRunReads('session-b'), b), b, {
      response: run('session-b'),
    })
    state = settleRunRead(beginRunRead(state, read('session-b', 3)), read('session-b', 3), {
      error: notFound(),
    })

    expect(state.notFound).toBe(true)
    expect(state.loading).toBe(false)
    expect(shownRunReads(state, 'session-b').response).toBeNull()
  })

  it('keeps the run on screen under a failed refresh, and drops it under a 404 refresh', () => {
    const first = read('session-b', 1)
    const state = settleRunRead(beginRunRead(initialRunReads('session-b'), first), first, {
      response: run('session-b'),
    })

    const refresh = read('session-b', 2)
    const failed = settleRunRead(beginRunRead(state, refresh), refresh, {
      error: new Error('boom'),
    })
    expect(shownSession(failed, 'session-b')).toBe('session-b')
    expect(failed.error).toBe('boom')
    expect(failed.loading).toBe(false)

    const gone = settleRunRead(beginRunRead(state, read('session-b', 3)), read('session-b', 3), {
      error: notFound(),
    })
    expect(gone.response).toBeNull()
    expect(gone.notFound).toBe(true)
  })

  it('keeps the newer read of one session when the older read answers last', () => {
    const older = read('session-b', 1)
    const newer = read('session-b', 2)
    let state = beginRunRead(initialRunReads('session-b'), older)
    state = beginRunRead(state, newer)
    state = settleRunRead(state, older, { response: run('old') })
    state = settleRunRead(state, newer, { response: run('session-b') })

    expect(shownSession(state, 'session-b')).toBe('session-b')
  })

  it('shows the skeleton, not the previous session, while the named session is unread', () => {
    const a = read('session-a', 1)
    const loaded = settleRunRead(beginRunRead(initialRunReads('session-a'), a), a, {
      response: run('session-a'),
    })

    const showing = shownRunReads(loaded, 'session-b')

    expect(showing.response).toBeNull()
    expect(showing.loading).toBe(true)
  })
})
