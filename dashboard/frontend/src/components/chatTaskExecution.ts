import type { Dispatch, SetStateAction } from 'react'

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
  type PlaygroundAttachment,
} from './playgroundFileAttachments'
import {
  playgroundErrorPresentation,
  type PlaygroundErrorInput,
} from './playgroundErrorPresentation'
import {
  assertPlaygroundResponseSuccess,
  consumePlaygroundResponseBody,
} from './chatTaskResponseSupport'
import { ChatTaskResponseState } from './chatTaskResponseState'
import { runToolLoop } from './chatTaskToolLoop'
import type { Message, PlaygroundTask } from './ChatComponentTypes'
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
  registerAbortController: (conversationId: string, controller: AbortController | null) => void
  setConversationError: (conversationId: string, error: PlaygroundErrorInput) => void
  setConversationThinking: (conversationId: string, visible: boolean) => void
  setExpandedToolCards: Dispatch<SetStateAction<Set<string>>>
  task: PlaygroundTask
  updateConversationMessages: UpdateConversationMessages
}

interface PreparedPlaygroundTask {
  attachments: PlaygroundAttachment[]
  exactMessages: OutboundChatMessage[]
  trimmedInput: string
}

interface PlaygroundExecutionRequest {
  activeTools: ToolDefinition[]
  chatMessages: OutboundChatMessage[]
  requestBody: Record<string, unknown>
}

interface PlaygroundExecutionRuntime {
  abortController: AbortController
  assistantMessageId: string
  responseState: ChatTaskResponseState
  timeoutHandle: ReturnType<typeof globalThis.setTimeout>
}

const preparePlaygroundTask = (task: PlaygroundTask): PreparedPlaygroundTask | null => {
  const trimmedInput = task.prompt.trim()
  const attachments = task.attachments ?? []
  const exactMessages = Array.isArray(task.exactRequest?.messages)
    ? (task.exactRequest.messages as OutboundChatMessage[])
    : []
  if (!trimmedInput && attachments.length === 0 && exactMessages.length === 0) {
    return null
  }
  return { attachments, exactMessages, trimmedInput }
}

const createUserMessage = (
  task: PlaygroundTask,
  preparedTask: PreparedPlaygroundTask,
  generateId: () => string,
): Message => {
  const attachmentImages = toPlaygroundAttachmentImages(preparedTask.attachments)
  const displayImages = [...(task.displayMessage?.images ?? []), ...attachmentImages]
  const exactUserContent = [...preparedTask.exactMessages]
    .reverse()
    .find((message) => message.role === 'user')?.content

  return {
    id: generateId(),
    role: 'user',
    content: task.displayMessage?.content ?? preparedTask.trimmedInput,
    images: displayImages.length > 0 ? displayImages : undefined,
    attachments:
      preparedTask.attachments.length > 0
        ? toPlaygroundAttachmentSummaries(preparedTask.attachments)
        : undefined,
    playgroundAttachments:
      preparedTask.attachments.length > 0 ? preparedTask.attachments : undefined,
    requestContent:
      exactUserContent ??
      buildPlaygroundUserContent(preparedTask.trimmedInput, preparedTask.attachments),
    timestamp: new Date(),
  }
}

const beginTaskExecution = (
  options: RunPlaygroundTaskOptions,
  preparedTask: PreparedPlaygroundTask,
): PlaygroundExecutionRuntime => {
  const { generateId, registerAbortController, setConversationError, task } = options
  setConversationError(task.conversationId, null)

  const assistantMessageId = generateId()
  const abortController = new AbortController()
  const timeoutHandle = globalThis.setTimeout(() => {
    abortController.abort(
      new DOMException(
        `Playground request timed out after ${PLAYGROUND_REQUEST_TIMEOUT_MS / 1000} seconds.`,
        'TimeoutError',
      ),
    )
  }, PLAYGROUND_REQUEST_TIMEOUT_MS)
  const userMessage = createUserMessage(task, preparedTask, generateId)
  const assistantMessage: Message = {
    id: assistantMessageId,
    role: 'assistant',
    content: '',
    timestamp: new Date(),
    isStreaming: true,
  }

  options.updateConversationMessages(task.conversationId, (prev) => [
    ...prev,
    ...(task.appendPromptMessage === false ? [] : [userMessage]),
    assistantMessage,
  ])
  registerAbortController(task.conversationId, abortController)
  options.setConversationThinking(task.conversationId, true)

  return {
    abortController,
    assistantMessageId,
    responseState: new ChatTaskResponseState({
      assistantMessageId,
      conversationId: task.conversationId,
      requestStartedAt: Date.now(),
      updateConversationMessages: options.updateConversationMessages,
    }),
    timeoutHandle,
  }
}

const buildExecutionRequest = (
  options: RunPlaygroundTaskOptions,
  preparedTask: PreparedPlaygroundTask,
): PlaygroundExecutionRequest => {
  const { buildTaskTools, clawManagementDisabled, getConversationMessagesSnapshot, task } = options
  const exactTools = Array.isArray(task.exactRequest?.tools)
    ? (task.exactRequest.tools as ToolDefinition[])
    : null
  const activeTools = task.exactRequest ? (exactTools ?? []) : buildTaskTools(task)
  const chatMessages = task.exactRequest
    ? preparedTask.exactMessages
    : buildChatMessages(
        getConversationMessagesSnapshot(task.conversationId),
        preparedTask.trimmedInput,
        task.requestOptions.enableClawMode && !clawManagementDisabled,
        preparedTask.attachments,
      )
  const requestBody = task.exactRequest
    ? buildExactChatRequestBody(task.exactRequest, task.requestOptions.model)
    : buildChatRequestBody(task.requestOptions.model, chatMessages, activeTools)

  return { activeTools, chatMessages, requestBody }
}

const completeTaskToolCalls = async (
  options: RunPlaygroundTaskOptions,
  runtime: PlaygroundExecutionRuntime,
  request: PlaygroundExecutionRequest,
): Promise<void> => {
  const { responseState } = runtime
  if (!responseState.hasToolCalls) return
  if (options.task.requestOptions.executeToolCalls === false) {
    responseState.skipPendingToolCalls()
    return
  }

  await runToolLoop({
    activeTools: request.activeTools,
    assistantMessageId: runtime.assistantMessageId,
    abortSignal: runtime.abortController.signal,
    endpoint: options.endpoint,
    executeTools: options.executeTools,
    expandedToolCardCount: options.expandedToolCardCount,
    initialMessages: [...request.chatMessages],
    latestThinkingProcessRef: responseState.latestThinkingProcessRef,
    mergeToolCallsIntoState: (parsedToolCalls, idPrefix, status) =>
      responseState.mergeToolCallsIntoState(parsedToolCalls, idPrefix, status),
    setExpandedToolCards: options.setExpandedToolCards,
    syncAssistantToolCalls: () => responseState.syncAssistantToolCalls(),
    task: options.task,
    toolCallsMap: responseState.toolCallsMap,
    updateConversationMessages: options.updateConversationMessages,
  })
}

const executePreparedTask = async (
  options: RunPlaygroundTaskOptions,
  preparedTask: PreparedPlaygroundTask,
  runtime: PlaygroundExecutionRuntime,
): Promise<void> => {
  const request = buildExecutionRequest(options, preparedTask)
  const response = await fetch(options.endpoint, {
    method: 'POST',
    headers: buildPlaygroundRequestHeaders(options.task.conversationId),
    body: JSON.stringify(request.requestBody),
    signal: runtime.abortController.signal,
  })

  await assertPlaygroundResponseSuccess(response)
  if (runtime.responseState.captureResponseHeaders(collectResponseHeaders(response))) {
    options.setConversationThinking(options.task.conversationId, false)
  }
  await consumePlaygroundResponseBody(response, (parsedCompletion, streaming) => {
    runtime.responseState.applyParsedCompletion(parsedCompletion, streaming)
  })
  if (request.activeTools.length > 0) {
    runtime.responseState.applyTextualToolCalls()
  }

  // Flush initial stream state before the tool loop so a scheduled empty
  // commit cannot overwrite the model's follow-up answer.
  runtime.responseState.drainStreamingChoiceSync()
  await completeTaskToolCalls(options, runtime, request)
  options.setConversationThinking(options.task.conversationId, false)
  runtime.responseState.finalize()
}

const handleTaskExecutionFailure = (
  options: RunPlaygroundTaskOptions,
  runtime: PlaygroundExecutionRuntime,
  error: unknown,
): void => {
  runtime.responseState.cancelStreamingChoiceSync()
  if (error instanceof Error && error.name === 'AbortError') return

  options.setConversationError(options.task.conversationId, playgroundErrorPresentation(error))
  options.updateConversationMessages(options.task.conversationId, (prev) =>
    prev.filter((message) => message.id !== runtime.assistantMessageId),
  )
}

const finishTaskExecution = (
  options: RunPlaygroundTaskOptions,
  runtime: PlaygroundExecutionRuntime,
): void => {
  globalThis.clearTimeout(runtime.timeoutHandle)
  runtime.responseState.cancelStreamingChoiceSync()
  options.setConversationThinking(options.task.conversationId, false)
  options.clearConversationActiveTask(options.task.conversationId, options.task.id)
  options.registerAbortController(options.task.conversationId, null)
}

export const runPlaygroundTask = async (options: RunPlaygroundTaskOptions): Promise<void> => {
  const preparedTask = preparePlaygroundTask(options.task)
  if (!preparedTask) return

  const runtime = beginTaskExecution(options, preparedTask)
  try {
    await executePreparedTask(options, preparedTask, runtime)
  } catch (error) {
    handleTaskExecutionFailure(options, runtime, error)
  } finally {
    finishTaskExecution(options, runtime)
  }
}
