import { afterEach, describe, expect, it, vi } from 'vitest'

import { runToolLoop } from './chatTaskToolLoop'
import type { Message, PlaygroundTask } from './ChatComponentTypes'
import type { ParsedToolCallChunk } from './chatResponseParsing'
import type { ToolCall, ToolDefinition } from '../tools'

const searchTool: ToolDefinition = {
  type: 'function',
  function: {
    name: 'search_web',
    description: 'Search the web.',
    parameters: { type: 'object', properties: {}, required: [] },
  },
}

const toolCallResponse = (index: number) =>
  new Response(
    JSON.stringify({
      choices: [
        {
          index: 0,
          finish_reason: 'tool_calls',
          message: {
            content: null,
            tool_calls: [
              {
                id: `call-${index}`,
                type: 'function',
                function: { name: 'search_web', arguments: '{}' },
              },
            ],
          },
        },
      ],
    }),
    { headers: { 'content-type': 'application/json' } },
  )

describe('runToolLoop', () => {
  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it('continues tool execution beyond six rounds until the model returns a final answer', async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(toolCallResponse(1))
      .mockResolvedValueOnce(toolCallResponse(2))
      .mockResolvedValueOnce(toolCallResponse(3))
      .mockResolvedValueOnce(toolCallResponse(4))
      .mockResolvedValueOnce(toolCallResponse(5))
      .mockResolvedValueOnce(toolCallResponse(6))
      .mockResolvedValueOnce(
        new Response(
          JSON.stringify({
            choices: [
              {
                index: 0,
                finish_reason: 'stop',
                message: { content: 'Final answer from the gathered evidence.' },
              },
            ],
          }),
          { headers: { 'content-type': 'application/json' } },
        ),
      )
    vi.stubGlobal('fetch', fetchMock)

    const initialToolCall: ToolCall = {
      id: 'call-0',
      type: 'function',
      function: { name: 'search_web', arguments: '{}' },
      status: 'pending',
    }
    const toolCallsMap = new Map<number, ToolCall>([[0, initialToolCall]])
    const task: PlaygroundTask = {
      id: 'task-1',
      conversationId: 'conversation-1',
      prompt: 'Research this deeply.',
      createdAt: 1,
      requestOptions: {
        enableClawMode: false,
        enableWebSearch: true,
        model: 'vllm-sr/blend',
      },
    }
    let messages: Message[] = [
      {
        id: 'assistant-1',
        role: 'assistant',
        content: '',
        timestamp: new Date(),
        isStreaming: true,
      },
    ]
    const executeTools = vi.fn(async (calls: ToolCall[]) =>
      calls.map((call) => ({ callId: call.id, name: call.function.name, content: 'evidence' })),
    )
    const mergeToolCallsIntoState = (
      parsedToolCalls: ParsedToolCallChunk[],
      _idPrefix: string,
      status: ToolCall['status'],
    ) => {
      parsedToolCalls.forEach((parsedToolCall) => {
        toolCallsMap.set(parsedToolCall.index, {
          id: parsedToolCall.id || `call-${parsedToolCall.index}`,
          type: 'function',
          function: {
            name: parsedToolCall.functionName || '',
            arguments: parsedToolCall.functionArguments || '',
          },
          status,
        })
      })
      return parsedToolCalls.length > 0
    }

    const result = await runToolLoop({
      activeTools: [searchTool],
      assistantMessageId: 'assistant-1',
      endpoint: '/api/router/v1/chat/completions',
      executeTools,
      expandedToolCardCount: 0,
      initialMessages: [{ role: 'user', content: 'Research this deeply.' }],
      latestThinkingProcessRef: { current: '' },
      mergeToolCallsIntoState,
      setExpandedToolCards: vi.fn(),
      syncAssistantToolCalls: vi.fn(),
      task,
      toolCallsMap,
      updateConversationMessages: (_conversationId, updater) => {
        messages = updater(messages)
      },
    })

    expect(result).toBe('Final answer from the gathered evidence.')
    expect(executeTools).toHaveBeenCalledTimes(7)
    expect(fetchMock).toHaveBeenCalledTimes(7)
    const [, finalRequest] = fetchMock.mock.calls[6] as [string, RequestInit]
    expect(JSON.parse(String(finalRequest.body))).toMatchObject({
      model: 'vllm-sr/blend',
      stream: true,
      tool_choice: 'auto',
    })
    expect(messages[0]).toMatchObject({
      content: 'Final answer from the gathered evidence.',
      toolCalls: expect.arrayContaining([
        expect.objectContaining({ id: 'call-6', status: 'completed' }),
      ]),
    })
  })
})
