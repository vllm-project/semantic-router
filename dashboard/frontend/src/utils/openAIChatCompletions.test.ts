import { afterEach, describe, expect, it, vi } from 'vitest'

import {
  collectOpenAIResponseHeaders,
  parseOpenAIChatCompletionChunk,
  streamOpenAIChatCompletion,
} from './openAIChatCompletions'

afterEach(() => vi.restoreAllMocks())

describe('OpenAI Chat Completions transport', () => {
  it('parses standard deltas and final usage', () => {
    expect(
      parseOpenAIChatCompletionChunk({
        id: 'chatcmpl-1',
        model: 'vllm-sr/blend',
        choices: [{ index: 0, delta: { content: 'Hello' }, finish_reason: null }],
      }),
    ).toMatchObject({ deltas: ['Hello'], responseId: 'chatcmpl-1', model: 'vllm-sr/blend' })

    expect(
      parseOpenAIChatCompletionChunk({
        choices: [],
        usage: { prompt_tokens: 10, completion_tokens: 4, total_tokens: 14 },
      }).usage,
    ).toEqual({ promptTokens: 10, completionTokens: 4, totalTokens: 14 })
  })

  it('only reveals safe routing and request metadata', () => {
    const headers = new Headers({
      'x-vsr-selected-model': 'local/model',
      'x-request-id': 'req-1',
      'set-cookie': 'secret=true',
    })
    expect(collectOpenAIResponseHeaders(headers)).toEqual({
      'x-request-id': 'req-1',
      'x-vsr-selected-model': 'local/model',
    })
  })

  it('streams the standard endpoint with delegated bearer authority', async () => {
    const body = [
      'data: {"id":"chatcmpl-1","model":"vllm-sr/blend","choices":[{"index":0,"delta":{"content":"He"}}]}',
      '',
      'data:{"choices":[{"index":0,"delta":{"content":"llo"},"finish_reason":"stop"}]}',
      '',
      'data: {"choices":[],"usage":{"prompt_tokens":2,"completion_tokens":1,"total_tokens":3}}',
      '',
      'data: [DONE]',
      '',
    ].join('\n')
    const fetchMock = vi.spyOn(globalThis, 'fetch').mockResolvedValue(
      new Response(body, {
        status: 200,
        headers: {
          'content-type': 'text/event-stream; charset=utf-8',
          'x-vsr-selected-decision': 'Simple',
        },
      }),
    )
    const deltas: string[] = []
    const order: string[] = []

    const result = await streamOpenAIChatCompletion({
      accessToken: 'delegated-secret',
      endpoint: '/v1/chat/completions',
      messages: [{ role: 'user', content: 'Hello' }],
      model: 'vllm-sr/blend',
      onDelta: (delta) => {
        deltas.push(delta)
        order.push(`delta:${delta}`)
      },
      sessionId: 'session-1',
      signal: new AbortController().signal,
    })
    order.push('complete')

    expect(deltas).toEqual(['He', 'llo'])
    expect(order).toEqual(['delta:He', 'delta:llo', 'complete'])
    expect(result).toMatchObject({
      finishReason: 'stop',
      model: 'vllm-sr/blend',
      responseId: 'chatcmpl-1',
      usage: { promptTokens: 2, completionTokens: 1, totalTokens: 3 },
    })
    const [, init] = fetchMock.mock.calls[0]
    expect(init).toMatchObject({ method: 'POST', credentials: 'omit', cache: 'no-store' })
    expect(new Headers(init?.headers).get('Authorization')).toBe('Bearer delegated-secret')
    const requestBody = JSON.parse(String(init?.body)) as Record<string, unknown>
    expect(requestBody).toMatchObject({
      model: 'vllm-sr/blend',
      stream: true,
      stream_options: { include_usage: true },
    })
    expect(requestBody).not.toHaveProperty('tools')
    expect(requestBody).not.toHaveProperty('tool_choice')
  })

  it('delivers network chunks as live deltas before final usage and headers resolve', async () => {
    let streamController!: ReadableStreamDefaultController<Uint8Array>
    const stream = new ReadableStream<Uint8Array>({
      start(controller) {
        streamController = controller
      },
    })
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(
      new Response(stream, {
        status: 200,
        headers: {
          'content-type': 'text/event-stream',
          'x-request-id': 'request-live-1',
          'x-vsr-selected-model': 'local/live',
        },
      }),
    )

    const encoder = new TextEncoder()
    const deltas: string[] = []
    let completed = false
    let resolveFirstDelta!: () => void
    const firstDelta = new Promise<void>((resolve) => {
      resolveFirstDelta = resolve
    })
    const completion = streamOpenAIChatCompletion({
      accessToken: 'delegated-secret',
      endpoint: '/v1/chat/completions',
      messages: [{ role: 'user', content: 'Stream this' }],
      model: 'vllm-sr/blend',
      onDelta: (delta) => {
        deltas.push(delta)
        resolveFirstDelta()
      },
      signal: new AbortController().signal,
    }).then((result) => {
      completed = true
      return result
    })

    streamController.enqueue(
      encoder.encode(
        'data: {"id":"chatcmpl-live","choices":[{"index":0,"delta":{"content":"First"}}]}\n\n',
      ),
    )
    await firstDelta
    expect(deltas).toEqual(['First'])
    expect(completed).toBe(false)

    streamController.enqueue(
      encoder.encode(
        'data: {"choices":[{"index":0,"delta":{"content":" second"},"finish_reason":"stop"}]}\n\n' +
          'data: {"choices":[],"usage":{"prompt_tokens":2,"completion_tokens":2,"total_tokens":4}}\n\n' +
          'data: [DONE]\n\n',
      ),
    )
    streamController.close()
    const result = await completion

    expect(deltas).toEqual(['First', ' second'])
    expect(result).toMatchObject({
      finishReason: 'stop',
      headers: {
        'x-request-id': 'request-live-1',
        'x-vsr-selected-model': 'local/live',
      },
      usage: { promptTokens: 2, completionTokens: 2, totalTokens: 4 },
    })
  })

  it('rejects a successful non-stream response instead of silently buffering it', async () => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(
      new Response('{"choices":[]}', {
        status: 200,
        headers: { 'content-type': 'application/json' },
      }),
    )

    await expect(
      streamOpenAIChatCompletion({
        accessToken: 'delegated-secret',
        endpoint: '/v1/chat/completions',
        messages: [{ role: 'user', content: 'Hello' }],
        model: 'vllm-sr/blend',
        onDelta: () => undefined,
        signal: new AbortController().signal,
      }),
    ).rejects.toThrow('did not return an OpenAI-compatible event stream')
  })

  it('parses fragmented CR-only frames and requires the terminal event', async () => {
    const encoder = new TextEncoder()
    const fragmented = new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(encoder.encode('data: {"choices":[{"delta":{"content":"Hel'))
        controller.enqueue(encoder.encode('lo"}}]}\r\rdata: [DO'))
        controller.enqueue(encoder.encode('NE]\r\r'))
        controller.close()
      },
    })
    const fetchMock = vi
      .spyOn(globalThis, 'fetch')
      .mockResolvedValueOnce(
        new Response(fragmented, { headers: { 'content-type': 'text/event-stream' } }),
      )
    const deltas: string[] = []
    await streamOpenAIChatCompletion({
      accessToken: 'delegated-secret',
      endpoint: '/v1/chat/completions',
      messages: [{ role: 'user', content: 'Hello' }],
      model: 'vllm-sr/blend',
      onDelta: (delta) => deltas.push(delta),
      signal: new AbortController().signal,
    })
    expect(deltas).toEqual(['Hello'])

    fetchMock.mockResolvedValueOnce(
      new Response('data: {"choices":[{"delta":{"content":"partial"}}]}\n\n', {
        headers: { 'content-type': 'text/event-stream' },
      }),
    )
    await expect(
      streamOpenAIChatCompletion({
        accessToken: 'delegated-secret',
        endpoint: '/v1/chat/completions',
        messages: [{ role: 'user', content: 'Hello' }],
        model: 'vllm-sr/blend',
        onDelta: () => undefined,
        signal: new AbortController().signal,
      }),
    ).rejects.toThrow('stream ended before completion')
  })
})
