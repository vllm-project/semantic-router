import { generatePlaygroundTaskId, type Message, type PlaygroundTask } from './ChatComponentTypes'
import {
  assertTrustedProbeImages,
  editableProbeMessageText,
  presentProbeMessage,
  replaceEditableProbeMessageText,
} from './playgroundStructuredContent'
import { buildPlaygroundUserContent, type PlaygroundAttachment } from './playgroundFileAttachments'
import type { PlaygroundInvocation } from '../types/playgroundInvocation'
import type { RecipeProbeMessage } from '../types/recipe'
import type { RouterModelOption } from '../utils/routerModelSelection'
import type { ToolCall, ToolResult } from '../tools'

export interface PreparedPlaygroundInvocation {
  key: string
  prompt: string
  promptDisplayText: string
  model?: string
  recipe: string
  history: RecipeProbeMessage[]
  messages: RecipeProbeMessage[]
  promptImages: Message['images']
  editableMessageIndex: number
  editable: boolean
  appendPromptMessage: boolean
  exactRequest: Record<string, unknown>
}

export const preparePlaygroundInvocation = (
  invocation: PlaygroundInvocation,
): PreparedPlaygroundInvocation => {
  const messages = invocation.messages.map((message) => ({ ...message }))
  assertTrustedProbeImages(messages)
  const terminalIndex = messages.length - 1
  const terminalIsUser = terminalIndex >= 0 && messages[terminalIndex]?.role === 'user'
  const messageIndex = terminalIsUser ? terminalIndex : -1
  const terminalMessage = terminalIsUser ? messages[messageIndex] : undefined
  const terminalPresentation = terminalMessage
    ? presentProbeMessage(terminalMessage)
    : { text: '', images: [] }
  const editableText = editableProbeMessageText(terminalMessage)
  const request = {
    ...invocation.request,
    messages,
    ...(invocation.tools ? { tools: invocation.tools } : {}),
    ...(invocation.toolChoice !== undefined ? { tool_choice: invocation.toolChoice } : {}),
    ...(invocation.model ? { model: invocation.model } : {}),
  }

  return {
    key: `${invocation.recipeDigest}:${invocation.probeId}:${invocation.intent}`,
    prompt: editableText ?? terminalPresentation.text,
    promptDisplayText: terminalPresentation.text,
    ...(invocation.model ? { model: invocation.model } : {}),
    recipe: invocation.recipe,
    history: terminalIsUser ? messages.slice(0, messageIndex) : messages,
    messages,
    promptImages: terminalPresentation.images,
    editableMessageIndex: messageIndex,
    editable: invocation.editable && terminalIsUser && editableText !== null,
    appendPromptMessage: terminalIsUser,
    exactRequest: request,
  }
}

export const resolveProbeRequestModel = (
  prepared: PreparedPlaygroundInvocation,
  routingModels: RouterModelOption[],
): string | undefined =>
  routingModels.find((candidate) => candidate.id === prepared.model)?.id ??
  routingModels.find((candidate) => candidate.recipe === prepared.recipe)?.id

export const materializeEditedProbeRequest = (
  prepared: PreparedPlaygroundInvocation,
  prompt: string,
  model: string,
  attachments: PlaygroundAttachment[] = [],
): Record<string, unknown> => {
  if (!prepared.editable || prepared.editableMessageIndex < 0) {
    throw new Error('This structured probe does not end in an editable user message.')
  }
  const editedContent = buildPlaygroundUserContent(prompt, attachments)
  const editedText =
    typeof editedContent === 'string'
      ? editedContent
      : (editedContent.find((part) => part.type === 'text')?.text ?? prompt)
  const addedImages =
    typeof editedContent === 'string'
      ? []
      : editedContent.filter((part) => part.type === 'image_url')
  const messages = prepared.messages.map((message, index) => {
    if (index !== prepared.editableMessageIndex) return { ...message }
    const replaced = replaceEditableProbeMessageText({ ...message, role: 'user' }, editedText)
    if (addedImages.length === 0) return replaced
    const content = Array.isArray(replaced.content)
      ? [...replaced.content, ...addedImages]
      : [{ type: 'text', text: replaced.content }, ...addedImages]
    return { ...replaced, content }
  })

  return {
    ...prepared.exactRequest,
    model,
    messages,
  }
}

export const createProbePlaygroundTask = (
  prepared: PreparedPlaygroundInvocation,
  conversationId: string,
  model: string,
): PlaygroundTask => ({
  id: generatePlaygroundTaskId(),
  conversationId,
  prompt: prepared.prompt,
  createdAt: Date.now(),
  requestOptions: {
    enableClawMode: false,
    enableWebSearch: false,
    executeToolCalls: false,
    model,
  },
  exactRequest: {
    ...prepared.exactRequest,
    model,
  },
  appendPromptMessage: prepared.appendPromptMessage,
  ...(prepared.appendPromptMessage
    ? {
        displayMessage: {
          content: prepared.promptDisplayText,
          ...(prepared.promptImages?.length ? { images: prepared.promptImages } : {}),
        },
      }
    : {}),
})

const asRecord = (value: unknown): Record<string, unknown> | null => {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return null
  return value as Record<string, unknown>
}

const toolCallsFromMessage = (message: RecipeProbeMessage): ToolCall[] => {
  if (!Array.isArray(message.tool_calls)) return []
  return message.tool_calls.flatMap((raw, index) => {
    const call = asRecord(raw)
    const fn = asRecord(call?.function)
    const name = typeof fn?.name === 'string' ? fn.name : ''
    if (!call || !name) return []
    const rawArguments = fn?.arguments
    const args =
      typeof rawArguments === 'string' ? rawArguments : JSON.stringify(rawArguments ?? {})
    return [
      {
        id: typeof call.id === 'string' && call.id ? call.id : `probe-tool-${index}`,
        type: 'function' as const,
        function: { name, arguments: args },
        status: 'pending' as const,
      },
    ]
  })
}

export const toPlaygroundHistory = (
  history: RecipeProbeMessage[],
  generateId: () => string,
): Message[] => {
  const result: Message[] = []
  const toolOwners = new Map<string, Message>()

  history.forEach((message) => {
    if (message.role === 'tool') {
      const callID = typeof message.tool_call_id === 'string' ? message.tool_call_id : ''
      const owner = toolOwners.get(callID)
      if (!owner) {
        result.push({
          id: generateId(),
          role: 'system',
          content: `Tool result${callID ? ` (${callID})` : ''}: ${presentProbeMessage(message).text}`,
          timestamp: new Date(),
        })
        return
      }
      const call = owner.toolCalls?.find((candidate) => candidate.id === callID)
      if (call) call.status = 'completed'
      const toolResult: ToolResult = {
        callId: callID,
        name: call?.function.name || 'tool',
        content: message.content,
      }
      owner.toolResults = [...(owner.toolResults ?? []), toolResult]
      return
    }

    if (message.role !== 'user' && message.role !== 'assistant' && message.role !== 'system') return
    const toolCalls = message.role === 'assistant' ? toolCallsFromMessage(message) : []
    const presentation = presentProbeMessage(message)
    const next: Message = {
      id: generateId(),
      role: message.role,
      content: presentation.text,
      requestContent: message.content,
      ...(presentation.images.length > 0 ? { images: presentation.images } : {}),
      timestamp: new Date(),
      ...(toolCalls.length > 0 ? { toolCalls } : {}),
    }
    result.push(next)
    toolCalls.forEach((call) => toolOwners.set(call.id, next))
  })

  return result
}
