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
})
