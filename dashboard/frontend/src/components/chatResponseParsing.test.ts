import { describe, expect, it } from 'vitest'

import { consumeEventStream, parseChatCompletionObject } from './chatResponseParsing'

describe('parseChatCompletionObject', () => {
  it('does not reuse a top-level candidate list as every Ratings choice model', () => {
    const parsed = parseChatCompletionObject({
      model: 'model-a,model-b',
      choices: [
        { index: 0, message: { content: 'a' } },
        { index: 1, message: { content: 'b' } },
      ],
    })

    expect(parsed?.choices.map((choice) => choice.model)).toEqual([undefined, undefined])
  })

  it('keeps an explicit per-choice model for Ratings feedback binding', () => {
    const parsed = parseChatCompletionObject({
      model: 'model-a,model-b',
      choices: [
        { index: 0, model: 'model-a', message: { content: 'a' } },
        { index: 1, model: 'model-b', message: { content: 'b' } },
      ],
    })

    expect(parsed?.choices.map((choice) => choice.model)).toEqual(['model-a', 'model-b'])
  })
})

describe('consumeEventStream', () => {
  it('accepts data lines without whitespace after the colon', async () => {
    const body = 'data:{"choices":[{"index":0,"message":{"content":"hello"}}]}\n\ndata: [DONE]\n\n'
    const stream = new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(new TextEncoder().encode(body))
        controller.close()
      },
    })
    const received: string[] = []

    await consumeEventStream(stream, (data) => received.push(data))

    expect(received).toEqual(['{"choices":[{"index":0,"message":{"content":"hello"}}]}'])
  })
})
