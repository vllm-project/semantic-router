import { afterEach, describe, expect, it, vi } from 'vitest'

import { runPlaygroundTask } from './chatTaskExecution'
import type { Message, PlaygroundTask } from './ChatComponentTypes'
import type { ToolDefinition } from '../tools'

const probeTool: ToolDefinition = {
  type: 'function',
  function: {
    name: 'lookup_policy',
    description: 'Look up a policy.',
    parameters: { type: 'object', properties: {}, required: [] },
  },
}

describe('runPlaygroundTask', () => {
  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it('sends the server materialized probe request without injecting or executing tools', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(
        JSON.stringify({
          choices: [
            {
              index: 0,
              message: {
                content: null,
                tool_calls: [
                  {
                    id: 'call-1',
                    type: 'function',
                    function: { name: 'lookup_policy', arguments: '{}' },
                  },
                ],
              },
            },
          ],
        }),
        { headers: { 'content-type': 'application/json' } },
      ),
    )
    vi.stubGlobal('fetch', fetchMock)

    const task: PlaygroundTask = {
      id: 'task-1',
      conversationId: 'conversation-1',
      prompt: 'Check the policy.',
      createdAt: 1,
      requestOptions: {
        enableClawMode: false,
        enableWebSearch: false,
        executeToolCalls: false,
        model: 'team/custom-balanced',
      },
      exactRequest: {
        model: 'team/custom-balanced',
        messages: [{ role: 'user', content: 'Check the policy.' }],
        tools: [probeTool],
        temperature: 0,
      },
    }
    const buildTaskTools = vi.fn(() => [probeTool])
    const executeTools = vi.fn(async () => [])
    let messages: Message[] = []
    let nextId = 0

    await runPlaygroundTask({
      buildTaskTools,
      clawManagementDisabled: false,
      clearConversationActiveTask: vi.fn(),
      endpoint: '/api/router/v1/chat/completions',
      executeTools,
      expandedToolCardCount: 0,
      generateId: () => `message-${nextId++}`,
      getConversationMessagesSnapshot: () => messages,
      getCurrentConversationId: () => 'conversation-1',
      registerAbortController: vi.fn(),
      setConversationError: vi.fn(),
      setConversationHeaderReveal: vi.fn(),
      setConversationThinking: vi.fn(),
      setExpandedToolCards: vi.fn(),
      task,
      updateConversationMessages: (_conversationId, updater) => {
        messages = updater(messages)
      },
    })

    expect(buildTaskTools).not.toHaveBeenCalled()
    expect(executeTools).not.toHaveBeenCalled()
    expect(fetchMock).toHaveBeenCalledOnce()
    const [, requestInit] = fetchMock.mock.calls[0] as [string, RequestInit]
    expect(JSON.parse(String(requestInit.body))).toEqual({
      model: 'team/custom-balanced',
      messages: [{ role: 'user', content: 'Check the policy.' }],
      tools: [probeTool],
      temperature: 0,
      stream: true,
      max_completion_tokens: 2048,
    })
    expect(requestInit.headers).toMatchObject({ 'x-session-id': 'conversation-1' })
    expect(messages[messages.length - 1]).toMatchObject({
      role: 'assistant',
      isStreaming: false,
      toolCalls: [{ status: 'skipped' }],
    })
  })

  it('keeps an edited probe image request exact while rendering both probe and uploaded previews', async () => {
    const probeImage = 'data:image/png;base64,AA=='
    const uploadedImage = 'data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///ywAAAAAAQABAAACAUwAOw=='
    const exactContent = [
      { type: 'text', text: 'Describe both images.' },
      { type: 'image_url', image_url: { url: probeImage } },
      { type: 'image_url', image_url: { url: uploadedImage } },
    ]
    const fetchMock = vi
      .fn()
      .mockResolvedValue(
        new Response(
          JSON.stringify({ choices: [{ index: 0, message: { content: 'Two images.' } }] }),
          { headers: { 'content-type': 'application/json' } },
        ),
      )
    vi.stubGlobal('fetch', fetchMock)

    const task: PlaygroundTask = {
      id: 'task-multimodal-edit',
      conversationId: 'conversation-multimodal-edit',
      prompt: 'Describe both images.',
      attachments: [
        {
          id: 'uploaded-image',
          fileName: 'uploaded.gif',
          sizeBytes: 43,
          kind: 'image',
          mediaType: 'image/gif',
          content: uploadedImage,
        },
      ],
      createdAt: 1,
      requestOptions: {
        enableClawMode: false,
        enableWebSearch: false,
        executeToolCalls: false,
        model: 'vllm-sr/mom-v1-blend',
      },
      exactRequest: {
        model: 'vllm-sr/mom-v1-blend',
        messages: [{ role: 'user', content: exactContent }],
      },
      displayMessage: {
        content: 'Describe both images.',
        images: [{ src: probeImage, alt: 'Probe image 1' }],
      },
    }
    let messages: Message[] = []
    let nextID = 0

    await runPlaygroundTask({
      buildTaskTools: () => [],
      clawManagementDisabled: false,
      clearConversationActiveTask: vi.fn(),
      endpoint: '/api/router/v1/chat/completions',
      executeTools: vi.fn(async () => []),
      expandedToolCardCount: 0,
      generateId: () => `message-${nextID++}`,
      getConversationMessagesSnapshot: () => messages,
      getCurrentConversationId: () => task.conversationId,
      registerAbortController: vi.fn(),
      setConversationError: vi.fn(),
      setConversationHeaderReveal: vi.fn(),
      setConversationThinking: vi.fn(),
      setExpandedToolCards: vi.fn(),
      task,
      updateConversationMessages: (_conversationID, updater) => {
        messages = updater(messages)
      },
    })

    const [, requestInit] = fetchMock.mock.calls[0] as [string, RequestInit]
    expect(JSON.parse(String(requestInit.body))).toMatchObject({
      model: 'vllm-sr/mom-v1-blend',
      messages: [{ role: 'user', content: exactContent }],
      stream: true,
    })
    expect(messages[0]).toMatchObject({
      role: 'user',
      content: 'Describe both images.',
      requestContent: exactContent,
      images: [
        { src: probeImage, alt: 'Probe image 1' },
        { src: uploadedImage, alt: 'uploaded.gif' },
      ],
    })
  })

  it('preserves a streamed follow-up answer when the initial tool stream finishes within one frame', async () => {
    const encoder = new TextEncoder()
    const streamResponse = (chunks: string[]) =>
      new Response(
        new ReadableStream({
          start(controller) {
            chunks.forEach((chunk) => controller.enqueue(encoder.encode(chunk)))
            controller.close()
          },
        }),
        { headers: { 'content-type': 'text/event-stream' } },
      )
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(
        streamResponse([
          'data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call-1","type":"function","function":{"name":"lookup_policy","arguments":"{}"}}]}}]}\n\n',
          'data: {"choices":[{"index":0,"finish_reason":"tool_calls"}]}\n\n',
          'data: [DONE]\n\n',
        ]),
      )
      .mockResolvedValueOnce(
        streamResponse([
          'data: {"choices":[{"index":0,"delta":{"content":"Policy found."}}]}\n\n',
          'data: [DONE]\n\n',
        ]),
      )
    vi.stubGlobal('fetch', fetchMock)

    const task: PlaygroundTask = {
      id: 'task-tool-loop',
      conversationId: 'conversation-tool-loop',
      prompt: 'Check the policy.',
      createdAt: 1,
      requestOptions: {
        enableClawMode: false,
        enableWebSearch: false,
        model: 'vllm-sr/auto',
      },
    }
    let messages: Message[] = []
    let nextId = 0

    await runPlaygroundTask({
      buildTaskTools: () => [probeTool],
      clawManagementDisabled: false,
      clearConversationActiveTask: vi.fn(),
      endpoint: '/api/router/v1/chat/completions',
      executeTools: vi.fn(async () => [
        { callId: 'call-1', name: 'lookup_policy', content: { found: true } },
      ]),
      expandedToolCardCount: 0,
      generateId: () => `message-${nextId++}`,
      getConversationMessagesSnapshot: () => messages,
      getCurrentConversationId: () => 'conversation-tool-loop',
      registerAbortController: vi.fn(),
      setConversationError: vi.fn(),
      setConversationHeaderReveal: vi.fn(),
      setConversationThinking: vi.fn(),
      setExpandedToolCards: vi.fn(),
      task,
      updateConversationMessages: (_conversationId, updater) => {
        messages = updater(messages)
      },
    })

    expect(fetchMock).toHaveBeenCalledTimes(2)
    expect(messages[messages.length - 1]).toMatchObject({
      role: 'assistant',
      content: 'Policy found.',
      isStreaming: false,
      toolCalls: [{ id: 'call-1', status: 'completed' }],
    })
  })
})
