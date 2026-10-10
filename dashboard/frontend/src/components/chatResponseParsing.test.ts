import { describe, expect, it } from 'vitest'

import { parseChatCompletionObject } from './chatResponseParsing'

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
