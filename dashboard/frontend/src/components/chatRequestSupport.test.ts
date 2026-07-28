import { describe, expect, it } from 'vitest'

import { collectResponseHeaders } from './chatRequestSupport'

function responseWithHeaders(headers: Record<string, string>): Response {
  return new Response(null, { headers })
}

describe('collectResponseHeaders', () => {
  it('collects the looper latency and token usage headers (#2694)', () => {
    const response = responseWithHeaders({
      'x-vsr-looper-latency-ms': '74',
      'x-vsr-looper-prompt-tokens': '16',
      'x-vsr-looper-completion-tokens': '101',
      'x-vsr-looper-total-tokens': '117',
    })

    const collected = collectResponseHeaders(response)

    expect(collected['x-vsr-looper-latency-ms']).toBe('74')
    expect(collected['x-vsr-looper-prompt-tokens']).toBe('16')
    expect(collected['x-vsr-looper-completion-tokens']).toBe('101')
    expect(collected['x-vsr-looper-total-tokens']).toBe('117')
  })

  it('ignores headers outside the allowlist', () => {
    const response = responseWithHeaders({ 'x-not-tracked': 'value' })

    expect(collectResponseHeaders(response)).toEqual({})
  })
})
