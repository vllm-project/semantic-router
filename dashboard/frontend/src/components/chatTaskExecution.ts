import type { Dispatch, SetStateAction } from 'react'

import {
  buildChoicesArray,
  consumeEventStream,
  getFirstChoice,
  isEventStreamContentType,
  mergeParsedChoices,
  parseChatCompletionPayload,
  type ChoiceAccumulator,
  type ParsedChatCompletion,
  type ParsedToolCallChunk,
} from './chatResponseParsing'
import {
  buildChatMessages,
  buildChatRequestBody,
  buildExactChatRequestBody,
  buildPlaygroundRequestHeaders,
  collectResponseHeaders,
  PLAYGROUND_REQUEST_TIMEOUT_MS,
  type OutboundChatMessage,
} from './chatRequestSupport'
import {
  buildPlaygroundUserContent,
  toPlaygroundAttachmentImages,
  toPlaygroundAttachmentSummaries,
} from './playgroundFileAttachments'
import { createFrameSyncController } from './chatStreamingFrameSync'
import { runToolLoop } from './chatTaskToolLoop'
import {
  extractTextToolCalls,
  mergeToolCallArgumentChunk,
  resolveAssistantContentUpdate,
} from './chatToolCallSupport'
import type { Choice, Message, PlaygroundTask, ReMoMRoundResponse } from './ChatComponentTypes'
import type { ToolCall, ToolDefinition, ToolResult } from '../tools'

type UpdateConversationMessages = (
  conversationId: string,
  updater: (prev: Message[]) => Message[],
) => void

type ExecuteTools = (
  toolCalls: ToolCall[],
  context: { signal?: AbortSignal },
) => Promise<ToolResult[]>

interface RunPlaygroundTaskOptions {
  buildTaskTools: (task: PlaygroundTask) => ToolDefinition[]
  clawManagementDisabled: boolean
  clearConversationActiveTask: (conversationId: string, taskId: string) => void
  endpoint: string
  executeTools: ExecuteTools
  expandedToolCardCount: number
  generateId: () => string
  getConversationMessagesSnapshot: (conversationId: string) => Message[]
  getCurrentConversationId: () => string
  registerAbortController: (conversationId: string, controller: AbortController | null) => void
  setConversationError: (conversationId: string, error: string | null) => void
  setConversationHeaderReveal: (
    conversationId: string,
    headers: Record<string, string> | null,
    visible?: boolean,
  ) => void
  setConversationThinking: (conversationId: string, visible: boolean) => void
  setExpandedToolCards: Dispatch<SetStateAction<Set<string>>>
  task: PlaygroundTask
  updateConversationMessages: UpdateConversationMessages
}

export const runPlaygroundTask = async ({
  buildTaskTools,
  clawManagementDisabled,
  clearConversationActiveTask,
  endpoint,
  executeTools,
  expandedToolCardCount,
  generateId,
  getConversationMessagesSnapshot,
  getCurrentConversationId,
  registerAbortController,
  setConversationError,
  setConversationHeaderReveal,
  setConversationThinking,
  setExpandedToolCards,
  task,
  updateConversationMessages,
}: RunPlaygroundTaskOptions): Promise<void> => {
  const trimmedInput = task.prompt.trim()
  const taskAttachments = task.attachments ?? []
  const taskAttachmentImages = toPlaygroundAttachmentImages(taskAttachments)
  const displayImages = [...(task.displayMessage?.images ?? []), ...taskAttachmentImages]
  const exactMessages = Array.isArray(task.exactRequest?.messages)
    ? (task.exactRequest.messages as OutboundChatMessage[])
    : []
  if (!trimmedInput && taskAttachments.length === 0 && exactMessages.length === 0) return

  setConversationError(task.conversationId, null)

  const assistantMessageId = generateId()
  const responseHeaders: Record<string, string> = {}
  const latestThinkingProcessRef = { current: '' }
  const abortController = new AbortController()
  const timeout = globalThis.setTimeout(() => {
    abortController.abort(
      new DOMException(
        `Playground request timed out after ${PLAYGROUND_REQUEST_TIMEOUT_MS / 1000} seconds.`,
        'TimeoutError',
      ),
    )
  }, PLAYGROUND_REQUEST_TIMEOUT_MS)
  const exactUserContent = [...exactMessages]
    .reverse()
    .find((message) => message.role === 'user')?.content
  const userMessage: Message = {
    id: generateId(),
    role: 'user',
    content: task.displayMessage?.content ?? trimmedInput,
    images: displayImages.length > 0 ? displayImages : undefined,
    attachments:
      taskAttachments.length > 0 ? toPlaygroundAttachmentSummaries(taskAttachments) : undefined,
    playgroundAttachments: taskAttachments.length > 0 ? taskAttachments : undefined,
    requestContent: exactUserContent ?? buildPlaygroundUserContent(trimmedInput, taskAttachments),
    timestamp: new Date(),
  }
  const assistantMessage: Message = {
    id: assistantMessageId,
    role: 'assistant',
    content: '',
    timestamp: new Date(),
    isStreaming: true,
  }

  updateConversationMessages(task.conversationId, (prev) => [
    ...prev,
    ...(task.appendPromptMessage === false ? [] : [userMessage]),
    assistantMessage,
  ])
  registerAbortController(task.conversationId, abortController)
  setConversationHeaderReveal(task.conversationId, null)
  setConversationThinking(task.conversationId, true)

  let cancelStreamingChoiceSync = () => {}

  try {
    const exactTools = Array.isArray(task.exactRequest?.tools)
      ? (task.exactRequest.tools as ToolDefinition[])
      : null
    const activeTools = task.exactRequest ? (exactTools ?? []) : buildTaskTools(task)
    const chatMessages = task.exactRequest
      ? exactMessages
      : buildChatMessages(
          getConversationMessagesSnapshot(task.conversationId),
          trimmedInput,
          task.requestOptions.enableClawMode && !clawManagementDisabled,
          taskAttachments,
        )
    const requestBody = task.exactRequest
      ? buildExactChatRequestBody(task.exactRequest, task.requestOptions.model)
      : buildChatRequestBody(task.requestOptions.model, chatMessages, activeTools)
    const response = await fetch(endpoint, {
      method: 'POST',
      headers: buildPlaygroundRequestHeaders(task.conversationId),
      body: JSON.stringify(requestBody),
      signal: abortController.signal,
    })

    if (!response.ok) {
      throw new Error(`API error: ${response.status} - ${await response.text()}`)
    }

    Object.assign(responseHeaders, collectResponseHeaders(response))
    if (Object.keys(responseHeaders).length > 0) {
      setConversationHeaderReveal(
        task.conversationId,
        responseHeaders,
        task.conversationId === getCurrentConversationId(),
      )
      setConversationThinking(task.conversationId, false)
    }

    const choiceContents: Map<number, ChoiceAccumulator> = new Map()
    const toolCallsMap: Map<number, ToolCall> = new Map()
    let hasToolCalls = false
    let isRatingsMode = false
    let reasoningMomResponses: ReMoMRoundResponse[] | undefined

    const syncAssistantToolCalls = () => {
      const currentToolCalls = Array.from(toolCallsMap.values())
      updateConversationMessages(task.conversationId, (prev) =>
        prev.map((message) =>
          message.id === assistantMessageId ? { ...message, toolCalls: currentToolCalls } : message,
        ),
      )
    }

    const mergeToolCallsIntoState = (
      parsedToolCalls: ParsedToolCallChunk[],
      idPrefix: string,
      status: ToolCall['status'],
    ): boolean => {
      if (parsedToolCalls.length === 0) return false
      hasToolCalls = true

      parsedToolCalls.forEach((parsedToolCall) => {
        const toolCallIndex = parsedToolCall.index
        if (!toolCallsMap.has(toolCallIndex)) {
          toolCallsMap.set(toolCallIndex, {
            id: parsedToolCall.id || `${idPrefix}-${toolCallIndex}`,
            type: 'function',
            function: {
              name: parsedToolCall.functionName || '',
              arguments: '',
            },
            status,
          })
        }

        const existingToolCall = toolCallsMap.get(toolCallIndex)!
        existingToolCall.status = status
        if (parsedToolCall.functionName) {
          existingToolCall.function.name = parsedToolCall.functionName
        }
        if (parsedToolCall.functionArguments) {
          existingToolCall.function.arguments = mergeToolCallArgumentChunk(
            existingToolCall.function.arguments,
            parsedToolCall.functionArguments,
          )
        }
        if (parsedToolCall.id) {
          existingToolCall.id = parsedToolCall.id
        }
      })

      return true
    }

    const commitAssistantChoices = (streaming: boolean) => {
      if (isRatingsMode) {
        const choicesArray = buildChoicesArray(choiceContents)
        const thinkingProcess =
          getFirstChoice(choiceContents)?.reasoningContent || latestThinkingProcessRef.current
        if (thinkingProcess) {
          latestThinkingProcessRef.current = thinkingProcess
        }

        updateConversationMessages(task.conversationId, (prev) =>
          prev.map((message) =>
            message.id === assistantMessageId
              ? {
                  ...message,
                  content: choicesArray[0]?.content || '',
                  choices: choicesArray,
                  thinkingProcess: thinkingProcess || message.thinkingProcess,
                  isStreaming: streaming,
                }
              : message,
          ),
        )
        return
      }

      const firstChoice = getFirstChoice(choiceContents)
      if (!firstChoice) return
      if (firstChoice.reasoningContent) {
        latestThinkingProcessRef.current = firstChoice.reasoningContent
      }

      updateConversationMessages(task.conversationId, (prev) =>
        prev.map((message) =>
          message.id === assistantMessageId
            ? {
                ...message,
                content: resolveAssistantContentUpdate(
                  message.content,
                  firstChoice.content,
                  hasToolCalls,
                ),
                thinkingProcess: firstChoice.reasoningContent || message.thinkingProcess,
                isStreaming: streaming,
              }
            : message,
        ),
      )
    }

    const streamingChoiceSync = createFrameSyncController(() => {
      commitAssistantChoices(true)
    })
    cancelStreamingChoiceSync = () => streamingChoiceSync.cancel()

    const syncAssistantChoices = (streaming: boolean) => {
      if (streaming) {
        streamingChoiceSync.schedule()
        return
      }

      streamingChoiceSync.cancel()
      commitAssistantChoices(false)
    }

    const applyParsedCompletion = (parsedCompletion: ParsedChatCompletion, streaming: boolean) => {
      if (parsedCompletion.reasoningMomResponses) {
        reasoningMomResponses = parsedCompletion.reasoningMomResponses
      }
      if (parsedCompletion.choices.length > 1) {
        isRatingsMode = true
      }
      mergeParsedChoices(choiceContents, parsedCompletion.choices)

      let shouldSyncToolCalls = false
      parsedCompletion.choices.forEach((parsedChoice) => {
        if (
          mergeToolCallsIntoState(parsedChoice.toolCalls, 'tool', streaming ? 'running' : 'pending')
        ) {
          shouldSyncToolCalls = true
        }
      })
      if (shouldSyncToolCalls) {
        syncAssistantToolCalls()
      }
      syncAssistantChoices(streaming)
    }

    if (!isEventStreamContentType(response.headers.get('content-type'))) {
      const parsedResponse = parseChatCompletionPayload(await response.text())
      if (!parsedResponse) throw new Error('Invalid JSON response')
      if (parsedResponse.errorMessage) throw new Error(parsedResponse.errorMessage)
      if (parsedResponse.choices.length === 0) throw new Error('No choices in response')
      applyParsedCompletion(parsedResponse, false)
    } else {
      if (!response.body) throw new Error('No response body')
      await consumeEventStream(response.body, (data) => {
        const parsedChunk = parseChatCompletionPayload(data)
        if (!parsedChunk) return
        if (parsedChunk.errorMessage) throw new Error(parsedChunk.errorMessage)
        applyParsedCompletion(parsedChunk, true)
      })
    }

    const firstChoice = getFirstChoice(choiceContents)
    if (firstChoice && activeTools.length > 0) {
      const textualToolCalls = extractTextToolCalls(firstChoice.content)
      if (textualToolCalls.toolCalls.length > 0) {
        firstChoice.content = textualToolCalls.content
        mergeToolCallsIntoState(textualToolCalls.toolCalls, 'text-tool', 'pending')
        syncAssistantToolCalls()
        syncAssistantChoices(false)
      }
    }

    // A fast stream can finish before the scheduled frame commits its initial
    // tool-call state. Flush it before the tool loop so that stale empty content
    // cannot overwrite the model's follow-up answer after tool execution.
    streamingChoiceSync.drain()

    if (hasToolCalls && task.requestOptions.executeToolCalls !== false) {
      await runToolLoop({
        activeTools,
        assistantMessageId,
        abortSignal: abortController.signal,
        endpoint,
        executeTools,
        expandedToolCardCount,
        initialMessages: [...chatMessages],
        latestThinkingProcessRef,
        mergeToolCallsIntoState,
        setExpandedToolCards,
        syncAssistantToolCalls,
        task,
        toolCallsMap,
        updateConversationMessages,
      })
    } else if (hasToolCalls) {
      toolCallsMap.forEach((toolCall) => {
        if (toolCall.status === 'pending' || toolCall.status === 'running') {
          toolCall.status = 'skipped'
        }
      })
      syncAssistantToolCalls()
    }

    const finalChoices: Choice[] | undefined = isRatingsMode
      ? buildChoicesArray(choiceContents)
      : undefined
    const finalThinkingProcess =
      latestThinkingProcessRef.current || getFirstChoice(choiceContents)?.reasoningContent || ''
    setConversationThinking(task.conversationId, false)
    streamingChoiceSync.drain()
    updateConversationMessages(task.conversationId, (prev) =>
      prev.map((message) =>
        message.id === assistantMessageId
          ? {
              ...message,
              isStreaming: false,
              headers: Object.keys(responseHeaders).length > 0 ? responseHeaders : undefined,
              choices: finalChoices,
              thinkingProcess: finalThinkingProcess || message.thinkingProcess,
              reasoning_mom_responses: reasoningMomResponses,
            }
          : message,
      ),
    )
  } catch (err) {
    cancelStreamingChoiceSync()
    if (err instanceof Error && err.name === 'AbortError') {
      return
    }
    setConversationError(task.conversationId, err instanceof Error ? err.message : 'Unknown error')
    updateConversationMessages(task.conversationId, (prev) =>
      prev.filter((message) => message.id !== assistantMessageId),
    )
  } finally {
    globalThis.clearTimeout(timeout)
    cancelStreamingChoiceSync()
    setConversationThinking(task.conversationId, false)
    clearConversationActiveTask(task.conversationId, task.id)
    registerAbortController(task.conversationId, null)
  }
}
