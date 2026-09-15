import {
  buildChoicesArray,
  getFirstChoice,
  mergeParsedChoices,
  type ChoiceAccumulator,
  type ParsedChatCompletion,
  type ParsedToolCallChunk,
} from './chatResponseParsing'
import { createFrameSyncController } from './chatStreamingFrameSync'
import {
  extractTextToolCalls,
  mergeToolCallArgumentChunk,
  resolveAssistantContentUpdate,
} from './chatToolCallSupport'
import type { Choice, Message, ReMoMRoundResponse } from './ChatComponentTypes'
import type { ToolCall } from '../tools'

type UpdateConversationMessages = (
  conversationId: string,
  updater: (prev: Message[]) => Message[],
) => void

interface ChatTaskResponseStateOptions {
  assistantMessageId: string
  conversationId: string
  requestStartedAt: number
  updateConversationMessages: UpdateConversationMessages
}

export class ChatTaskResponseState {
  readonly latestThinkingProcessRef = { current: '' }
  readonly toolCallsMap = new Map<number, ToolCall>()

  private readonly assistantMessageId: string
  private readonly choiceContents = new Map<number, ChoiceAccumulator>()
  private readonly conversationId: string
  private readonly requestStartedAt: number
  private readonly responseHeaders: Record<string, string> = {}
  private readonly streamingChoiceSync: ReturnType<typeof createFrameSyncController>
  private readonly updateConversationMessages: UpdateConversationMessages
  private firstResponseAt: number | null = null
  private hasAnyToolCalls = false
  private isRatingsMode = false
  private reasoningMomResponses: ReMoMRoundResponse[] | undefined

  constructor({
    assistantMessageId,
    conversationId,
    requestStartedAt,
    updateConversationMessages,
  }: ChatTaskResponseStateOptions) {
    this.assistantMessageId = assistantMessageId
    this.conversationId = conversationId
    this.requestStartedAt = requestStartedAt
    this.updateConversationMessages = updateConversationMessages
    this.streamingChoiceSync = createFrameSyncController(() => {
      this.commitAssistantChoices(true)
    })
  }

  get hasToolCalls(): boolean {
    return this.hasAnyToolCalls
  }

  captureResponseHeaders(headers: Record<string, string>): boolean {
    Object.assign(this.responseHeaders, headers)
    return Object.keys(this.responseHeaders).length > 0
  }

  cancelStreamingChoiceSync(): void {
    this.streamingChoiceSync.cancel()
  }

  drainStreamingChoiceSync(): void {
    this.streamingChoiceSync.drain()
  }

  applyParsedCompletion(parsedCompletion: ParsedChatCompletion, streaming: boolean): void {
    if (this.firstResponseAt === null && parsedCompletion.choices.length > 0) {
      this.firstResponseAt = Date.now()
      this.responseHeaders['x-vsr-ttft-ms'] = String(this.firstResponseAt - this.requestStartedAt)
    }
    if (parsedCompletion.reasoningMomResponses) {
      this.reasoningMomResponses = parsedCompletion.reasoningMomResponses
    }
    if (parsedCompletion.choices.length > 1) {
      this.isRatingsMode = true
    }
    mergeParsedChoices(this.choiceContents, parsedCompletion.choices)

    let shouldSyncToolCalls = false
    parsedCompletion.choices.forEach((parsedChoice) => {
      if (
        this.mergeToolCallsIntoState(
          parsedChoice.toolCalls,
          'tool',
          streaming ? 'running' : 'pending',
        )
      ) {
        shouldSyncToolCalls = true
      }
    })
    if (shouldSyncToolCalls) {
      this.syncAssistantToolCalls()
    }
    this.syncAssistantChoices(streaming)
  }

  applyTextualToolCalls(): void {
    const firstChoice = getFirstChoice(this.choiceContents)
    if (!firstChoice) return

    const textualToolCalls = extractTextToolCalls(firstChoice.content)
    if (textualToolCalls.toolCalls.length === 0) return

    firstChoice.content = textualToolCalls.content
    this.mergeToolCallsIntoState(textualToolCalls.toolCalls, 'text-tool', 'pending')
    this.syncAssistantToolCalls()
    this.syncAssistantChoices(false)
  }

  mergeToolCallsIntoState(
    parsedToolCalls: ParsedToolCallChunk[],
    idPrefix: string,
    status: ToolCall['status'],
  ): boolean {
    if (parsedToolCalls.length === 0) return false
    this.hasAnyToolCalls = true

    parsedToolCalls.forEach((parsedToolCall) => {
      const toolCallIndex = parsedToolCall.index
      if (!this.toolCallsMap.has(toolCallIndex)) {
        this.toolCallsMap.set(toolCallIndex, {
          id: parsedToolCall.id || `${idPrefix}-${toolCallIndex}`,
          type: 'function',
          function: {
            name: parsedToolCall.functionName || '',
            arguments: '',
          },
          status,
        })
      }

      const existingToolCall = this.toolCallsMap.get(toolCallIndex)
      if (!existingToolCall) return
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

  syncAssistantToolCalls(): void {
    const currentToolCalls = Array.from(this.toolCallsMap.values())
    this.updateConversationMessages(this.conversationId, (prev) =>
      prev.map((message) =>
        message.id === this.assistantMessageId
          ? { ...message, toolCalls: currentToolCalls }
          : message,
      ),
    )
  }

  skipPendingToolCalls(): void {
    this.toolCallsMap.forEach((toolCall) => {
      if (toolCall.status === 'pending' || toolCall.status === 'running') {
        toolCall.status = 'skipped'
      }
    })
    this.syncAssistantToolCalls()
  }

  finalize(): void {
    const finalChoices: Choice[] | undefined = this.isRatingsMode
      ? buildChoicesArray(this.choiceContents)
      : undefined
    const finalThinkingProcess =
      this.latestThinkingProcessRef.current ||
      getFirstChoice(this.choiceContents)?.reasoningContent ||
      ''
    this.responseHeaders['x-vsr-latency-ms'] = String(Date.now() - this.requestStartedAt)
    this.streamingChoiceSync.drain()
    this.updateConversationMessages(this.conversationId, (prev) =>
      prev.map((message) =>
        message.id === this.assistantMessageId
          ? {
              ...message,
              isStreaming: false,
              headers:
                Object.keys(this.responseHeaders).length > 0 ? this.responseHeaders : undefined,
              choices: finalChoices,
              thinkingProcess: finalThinkingProcess || message.thinkingProcess,
              reasoning_mom_responses: this.reasoningMomResponses,
            }
          : message,
      ),
    )
  }

  private syncAssistantChoices(streaming: boolean): void {
    if (streaming) {
      this.streamingChoiceSync.schedule()
      return
    }

    this.streamingChoiceSync.cancel()
    this.commitAssistantChoices(false)
  }

  private commitAssistantChoices(streaming: boolean): void {
    if (this.isRatingsMode) {
      this.commitRatingsChoices(streaming)
      return
    }
    this.commitFirstChoice(streaming)
  }

  private commitRatingsChoices(streaming: boolean): void {
    const choices = buildChoicesArray(this.choiceContents)
    const thinkingProcess =
      getFirstChoice(this.choiceContents)?.reasoningContent || this.latestThinkingProcessRef.current
    if (thinkingProcess) {
      this.latestThinkingProcessRef.current = thinkingProcess
    }

    this.updateConversationMessages(this.conversationId, (prev) =>
      prev.map((message) =>
        message.id === this.assistantMessageId
          ? {
              ...message,
              content: choices[0]?.content || '',
              choices,
              thinkingProcess: thinkingProcess || message.thinkingProcess,
              isStreaming: streaming,
            }
          : message,
      ),
    )
  }

  private commitFirstChoice(streaming: boolean): void {
    const firstChoice = getFirstChoice(this.choiceContents)
    if (!firstChoice) return
    if (firstChoice.reasoningContent) {
      this.latestThinkingProcessRef.current = firstChoice.reasoningContent
    }

    this.updateConversationMessages(this.conversationId, (prev) =>
      prev.map((message) =>
        message.id === this.assistantMessageId
          ? {
              ...message,
              content: resolveAssistantContentUpdate(
                message.content,
                firstChoice.content,
                this.hasAnyToolCalls,
              ),
              thinkingProcess: firstChoice.reasoningContent || message.thinkingProcess,
              isStreaming: streaming,
            }
          : message,
      ),
    )
  }
}
