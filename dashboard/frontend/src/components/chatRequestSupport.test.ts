import { describe, expect, it } from 'vitest'

import {
  buildChatMessages,
  buildChatRequestBody,
  buildExactChatRequestBody,
  buildPlaygroundRequestHeaders,
  collectResponseHeaders,
  PLAYGROUND_DEFAULT_MAX_COMPLETION_TOKENS,
  PLAYGROUND_MAX_REQUEST_BYTES,
} from './chatRequestSupport'

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

  it('preserves uploaded images as OpenAI multimodal content across turns', () => {
    const image = {
      id: 'image-1',
      fileName: 'architecture.png',
      sizeBytes: 68,
      content: 'data:image/png;base64,AAAA',
      kind: 'image' as const,
      mediaType: 'image/png',
    }
    const messages = buildChatMessages(
      [
        {
          id: 'user-image',
          role: 'user',
          content: 'What is shown here?',
          playgroundAttachments: [image],
          timestamp: new Date(),
        },
      ],
      'Compare it with this image.',
      false,
      [{ ...image, id: 'image-2', fileName: 'comparison.png' }],
    )

    expect(messages).toEqual([
      {
        role: 'user',
        content: [
          { type: 'text', text: 'What is shown here?' },
          { type: 'image_url', image_url: { url: image.content } },
        ],
      },
      {
        role: 'user',
        content: [
          { type: 'text', text: 'Compare it with this image.' },
          { type: 'image_url', image_url: { url: image.content } },
        ],
      },
    ])
  })

  it('replays canonical user and assistant image content from a probe history', () => {
    const userContent = [
      { type: 'text', text: 'Inspect this.' },
      { type: 'image_url', image_url: { url: 'data:image/png;base64,AA==' } },
    ]
    const assistantContent = [
      { type: 'output_text', text: 'It is a router diagram.' },
      { type: 'image_url', image_url: { url: 'data:image/png;base64,AQ==' } },
    ]
    const messages = buildChatMessages(
      [
        {
          id: 'user-probe',
          role: 'user',
          content: 'Inspect this.',
          requestContent: userContent,
          timestamp: new Date(),
        },
        {
          id: 'assistant-probe',
          role: 'assistant',
          content: 'It is a router diagram.',
          requestContent: assistantContent,
          timestamp: new Date(),
        },
      ],
      'What changed?',
      false,
    )

    expect(messages).toEqual([
      { role: 'user', content: userContent },
      { role: 'assistant', content: assistantContent },
      { role: 'user', content: 'What changed?' },
    ])
  })
})

describe('buildExactChatRequestBody', () => {
  it('preserves probe request fields while selecting the fallback model and streaming', () => {
    const messages = [{ role: 'user', content: 'route this request' }]
    const tools = [{ type: 'function', function: { name: 'lookup', parameters: {} } }]

    expect(
      buildExactChatRequestBody(
        {
          messages,
          tools,
          temperature: 0,
          stream: false,
        },
        'team/custom-balanced',
      ),
    ).toEqual({
      messages,
      tools,
      temperature: 0,
      model: 'team/custom-balanced',
      stream: true,
      max_completion_tokens: PLAYGROUND_DEFAULT_MAX_COMPLETION_TOKENS,
    })
  })

  it('keeps the model selected by the materialized probe request', () => {
    expect(
      buildExactChatRequestBody(
        {
          model: 'team/custom-private',
          messages: [{ role: 'user', content: 'private request' }],
        },
        'vllm-sr/auto',
      ).model,
    ).toBe('team/custom-private')
  })

  it('preserves an explicit probe completion budget', () => {
    const request = buildExactChatRequestBody(
      {
        messages: [{ role: 'user', content: 'briefly answer' }],
        max_tokens: 128,
      },
      'vllm-sr/auto',
    )

    expect(request.max_tokens).toBe(128)
    expect(request.max_completion_tokens).toBeUndefined()
  })
})

describe('playground request stability', () => {
  it('sets a bounded completion default for ordinary chat requests', () => {
    expect(buildChatRequestBody('vllm-sr/auto', [], [])).toMatchObject({
      max_completion_tokens: PLAYGROUND_DEFAULT_MAX_COMPLETION_TOKENS,
    })
  })

  it('rejects an encoded request larger than the Router request envelope', () => {
    expect(() =>
      buildExactChatRequestBody(
        {
          messages: [{ role: 'user', content: 'x'.repeat(PLAYGROUND_MAX_REQUEST_BYTES) }],
        },
        'vllm-sr/auto',
      ),
    ).toThrow('exceeds the 10 MB request limit')
  })

  it('does not grow a final run-plan request at the exact byte limit', () => {
    const emptyRequest = {
      model: 'vllm-sr/test',
      messages: [{ role: 'user', content: '' }],
      stream: true,
      max_completion_tokens: PLAYGROUND_DEFAULT_MAX_COMPLETION_TOKENS,
    }
    const emptyBytes = new TextEncoder().encode(JSON.stringify(emptyRequest)).byteLength
    const exactRequest = {
      ...emptyRequest,
      messages: [
        {
          role: 'user',
          content: 'x'.repeat(PLAYGROUND_MAX_REQUEST_BYTES - emptyBytes),
        },
      ],
    }

    expect(new TextEncoder().encode(JSON.stringify(exactRequest)).byteLength).toBe(
      PLAYGROUND_MAX_REQUEST_BYTES,
    )
    expect(buildExactChatRequestBody(exactRequest, 'vllm-sr/fallback')).toEqual(exactRequest)
    expect(() =>
      buildExactChatRequestBody(
        {
          ...exactRequest,
          messages: [{ role: 'user', content: `${exactRequest.messages[0].content}x` }],
        },
        'vllm-sr/fallback',
      ),
    ).toThrow('exceeds the 10 MB request limit')
  })

  it('pins every turn in one conversation to the same Router session', () => {
    expect(buildPlaygroundRequestHeaders('conv-demo')).toMatchObject({
      Accept: 'text/event-stream',
      'x-session-id': 'conv-demo',
      'x-vsr-debug': 'true',
    })
  })
})
