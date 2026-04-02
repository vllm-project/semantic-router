import type { Dispatch, MutableRefObject, SetStateAction } from 'react'

import {
  consumeEventStream,
  isEventStreamContentType,
  parseChatCompletionPayload,
  type ParsedChatCompletion,
  type ParsedToolCallChunk,
} from './chatResponseParsing'
import type { OutboundChatMessage } from './chatRequestSupport'
import type { Message, PlaygroundTask } from './ChatComponentTypes'
import type { ToolCall, ToolDefinition, ToolResult } from '../tools'
import { serializeToolResultForModel } from '../tools/toolResultSupport'

type UpdateConversationMessages = (
  conversationId: string,
  updater: (prev: Message[]) => Message[]
) => void

type ExecuteTools = (
  toolCalls: ToolCall[],
  context: { signal?: AbortSignal }
) => Promise<ToolResult[]>

interface RunToolLoopOptions {
  abortControllerRef: MutableRefObject<AbortController | null>
  activeTools: ToolDefinition[]
  assistantMessageId: string
  endpoint: string
  executeTools: ExecuteTools
  expandedToolCardCount: number
  initialMessages: OutboundChatMessage[]
  latestThinkingProcessRef: { current: string }
  mergeToolCallsIntoState: (
    parsedToolCalls: ParsedToolCallChunk[],
    idPrefix: string,
    status: ToolCall['status']
  ) => boolean
  setExpandedToolCards: Dispatch<SetStateAction<Set<string>>>
  syncAssistantToolCalls: () => void
  task: PlaygroundTask
  toolCallsMap: Map<number, ToolCall>
  updateConversationMessages: UpdateConversationMessages
}

export const runToolLoop = async ({
  abortControllerRef,
  activeTools,
  assistantMessageId,
  endpoint,
  executeTools,
  expandedToolCardCount,
  initialMessages,
  latestThinkingProcessRef,
  mergeToolCallsIntoState,
  setExpandedToolCards,
  syncAssistantToolCalls,
  task,
  toolCallsMap,
  updateConversationMessages,
}: RunToolLoopOptions): Promise<string> => {
  const MAX_TOOL_ITERATIONS = 30
  let iteration = 0
  let allToolCalls = Array.from(toolCallsMap.values())
  let allToolResults: ToolResult[] = []
  let finalContent = ''
  let currentMessages: OutboundChatMessage[] = [...initialMessages]

  while (iteration < MAX_TOOL_ITERATIONS) {
    iteration += 1
    const currentToolCalls = iteration === 1 ? allToolCalls : Array.from(toolCallsMap.values())
    if (currentToolCalls.length === 0) break

    currentToolCalls.forEach(tc => { tc.status = 'running' })
    const uiToolCalls = [...allToolCalls]
    if (iteration > 1) {
      currentToolCalls.forEach(tc => {
        if (!uiToolCalls.find(existingToolCall => existingToolCall.id === tc.id)) {
          uiToolCalls.push(tc)
        }
      })
      allToolCalls = uiToolCalls
    }

    updateConversationMessages(task.conversationId, prev =>
      prev.map(message =>
        message.id === assistantMessageId
          ? { ...message, toolCalls: [...uiToolCalls] }
          : message
      )
    )

    const toolResults = await executeTools(currentToolCalls, {
      signal: abortControllerRef.current?.signal,
    })

    toolResults.forEach(result => {
      const matchingToolCall = currentToolCalls.find(toolCall => toolCall.id === result.callId)
      if (matchingToolCall) {
        matchingToolCall.status = result.error ? 'failed' : 'completed'
      }
    })

    allToolResults = [...allToolResults, ...toolResults]
    updateConversationMessages(task.conversationId, prev =>
      prev.map(message =>
        message.id === assistantMessageId
          ? { ...message, toolCalls: [...uiToolCalls], toolResults: allToolResults }
          : message
      )
    )

    if (uiToolCalls.length > 0 && expandedToolCardCount === 0) {
      setExpandedToolCards(new Set([uiToolCalls[0].id]))
    }

    currentMessages = [
      ...currentMessages,
      {
        role: 'assistant',
        content: null,
        tool_calls: currentToolCalls.map(toolCall => ({
          id: toolCall.id,
          type: 'function',
          function: {
            name: toolCall.function.name,
            arguments: toolCall.function.arguments,
          },
        })),
      },
      ...toolResults.map(toolResult => ({
        role: 'tool',
        tool_call_id: toolResult.callId,
        content: serializeToolResultForModel(toolResult),
      })),
    ]

    const followUpResponse = await fetch(endpoint, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: task.requestOptions.model,
        messages: currentMessages,
        stream: true,
        tools: activeTools,
        tool_choice: 'auto',
      }),
      signal: abortControllerRef.current?.signal,
    })

    if (!followUpResponse.ok) {
      break
    }

    let followUpContent = ''
    let followUpThinking = ''
    let hasMoreToolCalls = false
    let streamFinishReason = ''
    toolCallsMap.clear()

    const syncFollowUpMessage = (streaming: boolean) => {
      if (followUpThinking) {
        latestThinkingProcessRef.current = followUpThinking
      }
      if (!followUpContent && !followUpThinking) return

      updateConversationMessages(task.conversationId, prev =>
        prev.map(message =>
          message.id === assistantMessageId
            ? {
              ...message,
              content: followUpContent || message.content,
              thinkingProcess: followUpThinking || message.thinkingProcess,
              isStreaming: streaming,
            }
            : message
        )
      )
    }

    const applyFollowUpCompletion = (parsedCompletion: ParsedChatCompletion, streaming: boolean) => {
      const resolvedFinishReason = parsedCompletion.choices[0]?.finishReason
      if (resolvedFinishReason) {
        streamFinishReason = resolvedFinishReason
      }

      let shouldSyncToolCalls = false
      for (const parsedChoice of parsedCompletion.choices) {
        if (parsedChoice.toolCalls.length > 0) {
          hasMoreToolCalls = true
          if (mergeToolCallsIntoState(parsedChoice.toolCalls, `tool-${iteration}`, 'pending')) {
            shouldSyncToolCalls = true
          }
        }
        if (parsedChoice.content) {
          followUpContent += parsedChoice.content
        }
        if (parsedChoice.reasoningContent) {
          followUpThinking += parsedChoice.reasoningContent
        }
      }

      if (!streamFinishReason) {
        streamFinishReason = hasMoreToolCalls ? 'tool_calls' : 'stop'
      }
      if (shouldSyncToolCalls) {
        syncAssistantToolCalls()
      }
      syncFollowUpMessage(streaming)
    }

    if (!isEventStreamContentType(followUpResponse.headers.get('content-type'))) {
      const parsedFollowUp = parseChatCompletionPayload(await followUpResponse.text())
      if (parsedFollowUp?.errorMessage) {
        break
      }
      if (parsedFollowUp && parsedFollowUp.choices.length > 0) {
        applyFollowUpCompletion(parsedFollowUp, false)
      }
    } else {
      if (!followUpResponse.body) break
      await consumeEventStream(followUpResponse.body, data => {
        const parsedFollowUpChunk = parseChatCompletionPayload(data)
        if (!parsedFollowUpChunk || parsedFollowUpChunk.errorMessage) {
          return
        }
        applyFollowUpCompletion(parsedFollowUpChunk, true)
      })
    }

    if (followUpContent) {
      finalContent = followUpContent
    }
    if (streamFinishReason === 'tool_calls' && toolCallsMap.size > 0) {
      continue
    }
    if (streamFinishReason === 'stop' || streamFinishReason === 'length' || !hasMoreToolCalls) {
      break
    }
  }

  if (finalContent) {
    updateConversationMessages(task.conversationId, prev =>
      prev.map(message =>
        message.id === assistantMessageId
          ? { ...message, content: finalContent }
          : message
      )
    )
    return finalContent
  }

  let fallbackContent = 'The model did not generate a response. Please try again.'
  if (allToolResults.length > 0) {
    const successResults = allToolResults.filter(result => !result.error)
    const failedResults = allToolResults.filter(result => result.error)
    if (successResults.length > 0) {
      fallbackContent = 'Based on the tool results, here is the relevant information:\n\n'
      successResults.forEach(result => {
        if (typeof result.content === 'string' && result.content.length > 0) {
          const summary = result.content.length > 500
            ? `${result.content.substring(0, 500)}...`
            : result.content
          fallbackContent += `${summary}\n\n`
        }
      })
    } else if (failedResults.length > 0) {
      fallbackContent = 'Some tool calls failed. Please try again or refine the request.'
    }
  }

  updateConversationMessages(task.conversationId, prev =>
    prev.map(message =>
      message.id === assistantMessageId
        ? { ...message, content: fallbackContent }
        : message
    )
  )
  return ''
}
