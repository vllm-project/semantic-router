import { describe, expect, it } from 'vitest'

import { buildChatMessages, collectResponseHeaders } from './chatRequestSupport'

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

describe('buildChatMessages', () => {
  it('preserves completed tool calls and results across user turns', () => {
    const messages = buildChatMessages(
      [
        {
          id: 'user-1',
          role: 'user',
          content: 'deep research on vllm-sr',
          timestamp: new Date(),
        },
        {
          id: 'assistant-1',
          role: 'assistant',
          content: 'The search results point to the architecture docs.',
          timestamp: new Date(),
          toolCalls: [
            {
              id: 'call-1',
              type: 'function',
              function: {
                name: 'search_web',
                arguments: '{"query":"vLLM Semantic Router"}',
              },
              status: 'completed',
            },
          ],
          toolResults: [
            {
              callId: 'call-1',
              name: 'search_web',
              content: { results: [{ title: 'Architecture' }] },
            },
          ],
        },
      ],
      'continue',
      false,
    )

    expect(messages.map((message) => message.role)).toEqual([
      'user',
      'assistant',
      'tool',
      'assistant',
      'user',
    ])
    expect(messages[1]).toMatchObject({
      content: null,
      tool_calls: [
        {
          id: 'call-1',
          function: {
            name: 'search_web',
            arguments: '{"query":"vLLM Semantic Router"}',
          },
        },
      ],
    })
    expect(messages[2]).toMatchObject({
      tool_call_id: 'call-1',
    })
    expect(messages[3].content).toBe('The search results point to the architecture docs.')
  })

  it('does not replay leaked textual tool markup into the next user turn', () => {
    const messages = buildChatMessages(
      [
        {
          id: 'assistant-legacy',
          role: 'assistant',
          content:
            'I will search now.\n<tool_call><function=search_web><parameter=query>vllm-sr</parameter></function></tool_call>',
          timestamp: new Date(),
        },
      ],
      'continue',
      false,
    )

    expect(messages).toEqual([
      { role: 'assistant', content: 'I will search now.' },
      { role: 'user', content: 'continue' },
    ])
  })
})
