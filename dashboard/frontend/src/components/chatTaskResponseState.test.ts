import { describe, expect, it } from 'vitest'

import { ChatTaskResponseState } from './chatTaskResponseState'
import { consumePlaygroundResponseBody } from './chatTaskResponseSupport'
import { parseChatCompletionObject } from './chatResponseParsing'
import { calculateTool } from '../tools/executors/calculate'

function responseState() {
  return new ChatTaskResponseState({
    assistantMessageId: 'assistant',
    conversationId: 'conversation',
    requestStartedAt: Date.now(),
    updateConversationMessages: () => {},
  })
}

describe('Playground tool argument transport', () => {
  it('preserves repeated digits, escaped text and interleaved tool calls across SSE byte boundaries', async () => {
    const state = responseState()
    const argumentsByIndex = [
      JSON.stringify({ expression: '997 * 991' }),
      JSON.stringify({ text: 'bookkeeper 😀😀 \\ path "quoted"', nested: [[11]] }),
    ]
    const chunks = argumentsByIndex.map((args) => [...args])
    const frames: string[] = []
    for (let offset = 0; offset < Math.max(...chunks.map((parts) => parts.length)); offset++) {
      for (let index = 0; index < chunks.length; index++) {
        if (chunks[index][offset] === undefined) continue
        frames.push(
          `data: ${JSON.stringify({
            choices: [
              {
                index: 0,
                delta: {
                  tool_calls: [
                    {
                      index,
                      ...(offset === 0 ? { id: `call-${index}`, type: 'function' } : {}),
                      function: {
                        ...(offset === 0 ? { name: index === 0 ? 'calculate' : 'echo' } : {}),
                        arguments: chunks[index][offset],
                      },
                    },
                  ],
                },
              },
            ],
          })}\n\n`,
        )
      }
    }
    frames.push(
      'data: {"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}\n\ndata: [DONE]\n\n',
    )
    const bytes = new TextEncoder().encode(frames.join(''))
    const response = new Response(
      new ReadableStream({
        start(controller) {
          for (let offset = 0; offset < bytes.length; offset += 7) {
            controller.enqueue(bytes.slice(offset, offset + 7))
          }
          controller.close()
        },
      }),
      { headers: { 'content-type': 'text/event-stream' } },
    )
    try {
      await consumePlaygroundResponseBody(response, (parsed, streaming) =>
        state.applyParsedCompletion(parsed, streaming),
      )
      for (let index = 0; index < argumentsByIndex.length; index++) {
        expect(state.toolCallsMap.get(index)?.function.arguments).toBe(argumentsByIndex[index])
        expect(state.toolCallsMap.get(index)?.id).toBe(`call-${index}`)
      }
      const args = JSON.parse(state.toolCallsMap.get(0)!.function.arguments)
      const result = await calculateTool.executor(args, {})
      expect(result.result).toBe(988027)
    } finally {
      state.cancelStreamingChoiceSync()
    }
  })

  it('replaces delta arguments with an explicit complete message snapshot without duplication', () => {
    const state = responseState()
    for (const value of [
      {
        delta: {
          tool_calls: [
            {
              index: 0,
              id: 'call-0',
              function: { name: 'calculate', arguments: '{"expression":"9' },
            },
          ],
        },
      },
      {
        message: {
          tool_calls: [
            {
              id: 'call-0',
              function: { name: 'calculate', arguments: '{"expression":"997 * 991"}' },
            },
          ],
        },
      },
    ]) {
      const parsed = parseChatCompletionObject({ choices: [{ index: 0, ...value }] })
      state.applyParsedCompletion(parsed!, false)
    }
    expect(state.toolCallsMap.get(0)?.function.arguments).toBe('{"expression":"997 * 991"}')
  })
})
