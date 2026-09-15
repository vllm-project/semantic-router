import { describe, expect, it, vi } from 'vitest'

import {
  assertPlaygroundResponseSuccess,
  consumePlaygroundResponseBody,
} from './chatTaskResponseSupport'

const eventStreamResponse = (events: string[]): Response => {
  const encoder = new TextEncoder()
  return new Response(
    new ReadableStream({
      start(controller) {
        events.forEach((event) => controller.enqueue(encoder.encode(`data: ${event}\n\n`)))
        controller.close()
      },
    }),
    { headers: { 'content-type': 'text/event-stream; charset=utf-8' } },
  )
}

describe('assertPlaygroundResponseSuccess', () => {
  it('constructs a stable product failure without hiding the technical response', async () => {
    const responseBody = 'worker://private-stack upstream=http://internal.example'
    const response = new Response(responseBody, {
      status: 503,
      statusText: 'Service Unavailable',
    })

    await expect(assertPlaygroundResponseSuccess(response)).rejects.toMatchObject({
      name: 'PlaygroundRequestFailure',
      productMessage: 'The model service is temporarily unavailable. Try again.',
      technicalDetails: `HTTP 503 Service Unavailable\n${responseBody}`,
    })
  })
})

describe('consumePlaygroundResponseBody', () => {
  it('dispatches a validated non-streaming completion', async () => {
    const applyParsedCompletion = vi.fn()
    const response = new Response(
      JSON.stringify({ choices: [{ index: 0, message: { content: 'Hello.' } }] }),
      { headers: { 'content-type': 'application/json' } },
    )

    await consumePlaygroundResponseBody(response, applyParsedCompletion)

    expect(applyParsedCompletion).toHaveBeenCalledOnce()
    expect(applyParsedCompletion).toHaveBeenCalledWith(
      expect.objectContaining({
        choices: [expect.objectContaining({ index: 0, content: 'Hello.' })],
      }),
      false,
    )
  })

  it('rejects malformed non-streaming JSON at the response boundary', async () => {
    const response = new Response('{not-json', {
      headers: { 'content-type': 'application/json' },
    })

    await expect(consumePlaygroundResponseBody(response, vi.fn())).rejects.toMatchObject({
      productMessage: 'The model service returned an invalid response. Try again.',
      technicalDetails: 'The response body was not valid chat-completion JSON.',
    })
  })

  it('rejects a non-streaming completion without choices', async () => {
    const response = new Response(JSON.stringify({ choices: [] }), {
      headers: { 'content-type': 'application/json' },
    })

    await expect(consumePlaygroundResponseBody(response, vi.fn())).rejects.toMatchObject({
      productMessage: 'The model service returned an incomplete response. Try again.',
      technicalDetails: 'The chat-completion response contained no choices.',
    })
  })

  it('preserves a model error as technical detail behind the product message', async () => {
    const response = new Response(
      JSON.stringify({ error: { message: 'backend model failed to load' } }),
      { headers: { 'content-type': 'application/json' } },
    )

    await expect(consumePlaygroundResponseBody(response, vi.fn())).rejects.toMatchObject({
      productMessage:
        'The model service could not complete this request. Review the model settings, then try again.',
      technicalDetails: 'backend model failed to load',
    })
  })

  it('rejects a streaming response without a readable body', async () => {
    const response = new Response(null, {
      headers: { 'content-type': 'text/event-stream' },
    })

    await expect(consumePlaygroundResponseBody(response, vi.fn())).rejects.toMatchObject({
      productMessage: 'The model service returned an incomplete response. Try again.',
      technicalDetails: 'The streaming response did not contain a response body.',
    })
  })

  it('ignores malformed stream events and dispatches valid completion chunks', async () => {
    const applyParsedCompletion = vi.fn()
    const response = eventStreamResponse([
      'not-json',
      '{"choices":[{"index":0,"delta":{"content":"Hello."}}]}',
      '[DONE]',
    ])

    await consumePlaygroundResponseBody(response, applyParsedCompletion)

    expect(applyParsedCompletion).toHaveBeenCalledOnce()
    expect(applyParsedCompletion).toHaveBeenCalledWith(
      expect.objectContaining({
        choices: [expect.objectContaining({ index: 0, content: 'Hello.' })],
      }),
      true,
    )
  })

  it('stops a stream when the service emits an error payload', async () => {
    const response = eventStreamResponse(['{"error":{"message":"stream worker disconnected"}}'])

    await expect(consumePlaygroundResponseBody(response, vi.fn())).rejects.toMatchObject({
      productMessage:
        'The model service could not complete this request. Review the model settings, then try again.',
      technicalDetails: 'stream worker disconnected',
    })
  })
  it.each(['application/json', 'text/event-stream'])(
    'rejects budget exhaustion in %s while dispatching the partial response',
    async (contentType) => {
      const apply = vi.fn()
      const choice = {
        index: 0,
        message: { content: 'Partial answer.', reasoning_content: 'Thinking.' },
        finish_reason: 'length',
      }
      const payload = JSON.stringify({ choices: [choice] })
      const response =
        contentType === 'application/json'
          ? new Response(payload, { headers: { 'content-type': contentType } })
          : eventStreamResponse([payload, '[DONE]'])

      await expect(consumePlaygroundResponseBody(response, apply)).rejects.toMatchObject({
        productMessage: expect.stringContaining('backend output limit was reached'),
        technicalDetails: expect.stringContaining('finish_reason: length'),
      })
      expect(apply).toHaveBeenCalledWith(
        expect.objectContaining({
          choices: [expect.objectContaining({ content: 'Partial answer.' })],
        }),
        contentType === 'text/event-stream',
      )
    },
  )

  it.each(['', 'Thinking without a final answer.'])(
    'rejects empty or reasoning-only successful responses (%s)',
    async (reasoning) => {
      const response = eventStreamResponse([
        JSON.stringify({
          choices: [{ index: 0, delta: { content: null, reasoning_content: reasoning } }],
        }),
        JSON.stringify({ choices: [{ index: 0, delta: {}, finish_reason: 'stop' }] }),
        '[DONE]',
      ])
      await expect(consumePlaygroundResponseBody(response, vi.fn())).rejects.toMatchObject({
        productMessage: expect.stringContaining(
          reasoning ? 'without a final answer' : 'no answer or tool calls',
        ),
      })
    },
  )

  it('accepts reasoning followed by a final answer and ignores usage-only trailer chunks', async () => {
    const response = eventStreamResponse([
      '{"choices":[{"index":0,"delta":{"reasoning_content":"Thinking."}}]}',
      '{"choices":[{"index":0,"delta":{"content":"Final answer."}}]}',
      '{"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}',
      '{"choices":[],"usage":{"completion_tokens":3923}}',
      '[DONE]',
    ])
    await expect(consumePlaygroundResponseBody(response, vi.fn())).resolves.toBeUndefined()
  })

  it('accepts a tool call with no answer text', async () => {
    const response = new Response(
      JSON.stringify({
        choices: [
          {
            index: 0,
            message: {
              content: null,
              tool_calls: [{ id: 'call-1', function: { name: 'lookup', arguments: '{}' } }],
            },
            finish_reason: 'tool_calls',
          },
        ],
      }),
      { headers: { 'content-type': 'application/json' } },
    )
    await expect(consumePlaygroundResponseBody(response, vi.fn())).resolves.toBeUndefined()
  })

  it('rejects a stream containing only malformed events', async () => {
    await expect(
      consumePlaygroundResponseBody(eventStreamResponse(['not-json', '[DONE]']), vi.fn()),
    ).rejects.toMatchObject({
      technicalDetails: 'The response stream contained no completion choices.',
    })
  })
  it('does not let a valid choice hide an empty alternative', async () => {
    const response = new Response(
      JSON.stringify({
        choices: [
          { index: 0, message: { content: 'Answer.' }, finish_reason: 'stop' },
          {
            index: 1,
            message: { content: null, reasoning_content: 'Thinking.' },
            finish_reason: 'stop',
          },
        ],
      }),
      { headers: { 'content-type': 'application/json' } },
    )
    await expect(consumePlaygroundResponseBody(response, vi.fn())).rejects.toMatchObject({
      technicalDetails: 'Completion choice 1 contained no non-empty answer text or tool calls.',
    })
  })

  it.each(['message', 'delta'])(
    'renders a refusal in %s as a valid model response',
    async (field) => {
      const apply = vi.fn()
      const payload = JSON.stringify({
        choices: [
          {
            index: 0,
            [field]: { content: null, refusal: 'I cannot help with that request.' },
            finish_reason: 'stop',
          },
        ],
      })
      const response =
        field === 'message'
          ? new Response(payload, { headers: { 'content-type': 'application/json' } })
          : eventStreamResponse([payload, '[DONE]'])
      await consumePlaygroundResponseBody(response, apply)
      expect(apply).toHaveBeenCalledWith(
        expect.objectContaining({
          choices: [expect.objectContaining({ content: 'I cannot help with that request.' })],
        }),
        field === 'delta',
      )
    },
  )
})
