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
    vi.useRealTimers()
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
        max_completion_tokens: 512,
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
      registerAbortController: vi.fn(),
      setConversationError: vi.fn(),
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
      max_completion_tokens: 512,
      stream: true,
    })
    expect(requestInit.headers).toMatchObject({
      'x-session-id': 'conversation-1',
      'x-conversation-id': 'conversation-1',
    })
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
      registerAbortController: vi.fn(),
      setConversationError: vi.fn(),
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

  it.each(['conversation-tool-loop', 'another-conversation'])(
    'preserves the streamed answer and conversation identity across both fetches for %s',
    async (conversationId) => {
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
        conversationId,
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
        registerAbortController: vi.fn(),
        setConversationError: vi.fn(),
        setConversationThinking: vi.fn(),
        setExpandedToolCards: vi.fn(),
        task,
        updateConversationMessages: (_conversationId, updater) => {
          messages = updater(messages)
        },
      })

      expect(fetchMock).toHaveBeenCalledTimes(2)
      for (const [, requestInit] of fetchMock.mock.calls as [string, RequestInit][]) {
        expect(requestInit.headers).toMatchObject({
          'x-session-id': conversationId,
          'x-conversation-id': conversationId,
        })
      }
      expect(messages[messages.length - 1]).toMatchObject({
        role: 'assistant',
        content: 'Policy found.',
        isStreaming: false,
        toolCalls: [{ id: 'call-1', status: 'completed' }],
      })
    },
  )

  it('separates an actionable HTTP failure from the raw response body', async () => {
    const rawResponse = 'worker://private-stack upstream=http://internal.example'
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(
        new Response(rawResponse, {
          status: 503,
          statusText: 'Service Unavailable',
        }),
      ),
    )
    const task: PlaygroundTask = {
      id: 'task-http-failure',
      conversationId: 'conversation-http-failure',
      prompt: 'Hello',
      createdAt: 1,
      requestOptions: {
        enableClawMode: false,
        enableWebSearch: false,
        model: 'vllm-sr/auto',
      },
    }
    const setConversationError = vi.fn()
    let messages: Message[] = []

    await runPlaygroundTask({
      buildTaskTools: () => [],
      clawManagementDisabled: false,
      clearConversationActiveTask: vi.fn(),
      endpoint: '/api/router/v1/chat/completions',
      executeTools: vi.fn(async () => []),
      expandedToolCardCount: 0,
      generateId: () => crypto.randomUUID(),
      getConversationMessagesSnapshot: () => messages,
      registerAbortController: vi.fn(),
      setConversationError,
      setConversationThinking: vi.fn(),
      setExpandedToolCards: vi.fn(),
      task,
      updateConversationMessages: (_conversationId, updater) => {
        messages = updater(messages)
      },
    })

    const lastFailureCall =
      setConversationError.mock.calls[setConversationError.mock.calls.length - 1]
    const failure = lastFailureCall?.[1]
    expect(failure).toEqual({
      message: 'The model service is temporarily unavailable. Try again.',
      technicalDetails: `HTTP 503 Service Unavailable\n${rawResponse}`,
    })
    expect(failure.message).not.toContain(rawResponse)
    expect(messages).toHaveLength(1)
  })
  it.each([false, true])(
    'preserves and marks truncated output (tool follow-up: %s)',
    async (toolFollowUp) => {
      const response = (payloads: unknown[]) => {
        const encoder = new TextEncoder()
        return new Response(
          new ReadableStream({
            start(controller) {
              payloads.forEach((payload) =>
                controller.enqueue(encoder.encode(`data: ${JSON.stringify(payload)}\n\n`)),
              )
              controller.close()
            },
          }),
          { headers: { 'content-type': 'text/event-stream' } },
        )
      }
      const fetchMock = vi.fn()
      if (toolFollowUp) {
        fetchMock.mockResolvedValueOnce(
          response([
            {
              choices: [
                {
                  index: 0,
                  delta: {
                    tool_calls: [
                      {
                        index: 0,
                        id: 'call-1',
                        function: { name: 'lookup_policy', arguments: '{}' },
                      },
                    ],
                  },
                  finish_reason: 'tool_calls',
                },
              ],
            },
          ]),
        )
      }
      fetchMock.mockResolvedValueOnce(
        response([
          {
            choices: [
              {
                index: 0,
                delta: { content: 'Partial answer.', reasoning_content: 'Partial reasoning.' },
              },
            ],
          },
          { choices: [{ index: 0, delta: {}, finish_reason: 'length' }] },
        ]),
      )
      vi.stubGlobal('fetch', fetchMock)
      const task: PlaygroundTask = {
        id: 'task-incomplete',
        conversationId: 'conversation-incomplete',
        prompt: 'Explain.',
        createdAt: 1,
        requestOptions: {
          enableClawMode: false,
          enableWebSearch: false,
          model: 'vllm-sr/balance',
        },
      }
      let messages: Message[] = []
      const setConversationError = vi.fn()
      const executeTools = vi.fn(async () => [
        { callId: 'call-1', name: 'lookup_policy', content: 'Evidence.' },
      ])
      await runPlaygroundTask({
        buildTaskTools: () => [probeTool],
        clawManagementDisabled: false,
        clearConversationActiveTask: vi.fn(),
        endpoint: '/api/router/v1/chat/completions',
        executeTools,
        expandedToolCardCount: 0,
        generateId: () => crypto.randomUUID(),
        getConversationMessagesSnapshot: () => messages,
        registerAbortController: vi.fn(),
        setConversationError,
        setConversationThinking: vi.fn(),
        setExpandedToolCards: vi.fn(),
        task,
        updateConversationMessages: (_id, updater) => {
          messages = updater(messages)
        },
      })
      expect(messages[messages.length - 1]).toMatchObject({
        role: 'assistant',
        content: 'Partial answer.',
        thinkingProcess: 'Partial reasoning.',
        isStreaming: false,
        incomplete: expect.stringContaining('backend output limit was reached'),
      })
      expect(setConversationError).toHaveBeenLastCalledWith(
        task.conversationId,
        expect.objectContaining({
          message: expect.stringContaining('backend output limit was reached'),
        }),
      )
      expect(executeTools).toHaveBeenCalledTimes(toolFollowUp ? 1 : 0)
      if (toolFollowUp) {
        expect(messages[messages.length - 1]?.toolCalls).toEqual([
          expect.objectContaining({ id: 'call-1', status: 'completed' }),
        ])
      }
      for (const [, requestInit] of fetchMock.mock.calls) {
        expect(JSON.parse(requestInit.body)).not.toHaveProperty('max_completion_tokens')
        expect(JSON.parse(requestInit.body)).not.toHaveProperty('max_tokens')
      }
    },
  )
  it.each(['complete', 'cancel'])(
    'does not impose a chat deadline and can %s after the old timeout',
    async (outcome) => {
      vi.useFakeTimers()
      let completeResponse: (response: Response) => void = () => undefined
      const requestState: { signal?: AbortSignal } = {}
      const fetchMock = vi.fn((_endpoint: string, request: RequestInit) => {
        requestState.signal = request.signal as AbortSignal
        return new Promise<Response>((resolve, reject) => {
          completeResponse = resolve
          requestState.signal?.addEventListener('abort', () =>
            reject(new DOMException('User stopped generation.', 'AbortError')),
          )
        })
      })
      vi.stubGlobal('fetch', fetchMock)
      const task: PlaygroundTask = {
        id: 'long-chat',
        conversationId: 'long-conversation',
        prompt: 'Think carefully.',
        createdAt: 1,
        requestOptions: { enableClawMode: false, enableWebSearch: false, model: 'vllm-sr/balance' },
      }
      let messages: Message[] = []
      const controllers: AbortController[] = []
      const setConversationError = vi.fn()
      const clearConversationActiveTask = vi.fn()
      const execution = runPlaygroundTask({
        buildTaskTools: () => [],
        clawManagementDisabled: false,
        clearConversationActiveTask,
        endpoint: '/api/router/v1/chat/completions',
        executeTools: vi.fn(async () => []),
        expandedToolCardCount: 0,
        generateId: () => crypto.randomUUID(),
        getConversationMessagesSnapshot: () => messages,
        registerAbortController: (_conversationId, controller) => {
          if (controller) controllers.push(controller)
        },
        setConversationError,
        setConversationThinking: vi.fn(),
        setExpandedToolCards: vi.fn(),
        task,
        updateConversationMessages: (_id, updater) => {
          messages = updater(messages)
        },
      })
      await vi.advanceTimersByTimeAsync(600_000)
      expect(requestState.signal?.aborted).toBe(false)
      const requestBody = JSON.parse(String(fetchMock.mock.calls[0][1].body))
      expect(requestBody).not.toHaveProperty('max_tokens')
      expect(requestBody).not.toHaveProperty('max_completion_tokens')
      if (outcome === 'cancel') controllers[0].abort()
      else
        completeResponse(
          new Response(
            JSON.stringify({
              choices: [
                {
                  index: 0,
                  message: { content: 'Completed after thinking.' },
                  finish_reason: 'stop',
                },
              ],
            }),
            { headers: { 'content-type': 'application/json' } },
          ),
        )
      await execution
      if (outcome === 'cancel') expect(requestState.signal?.aborted).toBe(true)
      else
        expect(messages[messages.length - 1]).toMatchObject({
          content: 'Completed after thinking.',
          isStreaming: false,
        })
      expect(setConversationError).toHaveBeenCalledTimes(1)
      expect(clearConversationActiveTask).toHaveBeenCalledWith(task.conversationId, task.id)
    },
  )
})
