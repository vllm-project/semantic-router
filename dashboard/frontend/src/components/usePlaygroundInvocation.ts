import { useEffect, useRef } from 'react'

import {
  createProbePlaygroundTask,
  preparePlaygroundInvocation,
  toPlaygroundHistory,
  type PreparedPlaygroundInvocation,
} from './playgroundInvocationSupport'
import {
  generateConversationId,
  generateMessageId,
  type Message,
  type PlaygroundTask,
} from './ChatComponentTypes'
import type { PlaygroundInvocation } from '../types/playgroundInvocation'
import type { RouterModelOption } from '../utils/routerModelSelection'

interface UsePlaygroundInvocationOptions {
  invocation?: PlaygroundInvocation | null
  isRoutingModelReady: boolean
  onInvocationConsumed?: () => void
  routingModels: RouterModelOption[]
  activateConversation: (conversationId: string, messages: Message[]) => void
  focusComposer: () => void
  setConversationError: (conversationId: string, error: string | null) => void
  setDraft: (draft: ActivePlaygroundInvocationDraft | null) => void
  setInputValue: (value: string) => void
  setModel: (model: string) => void
  startTask: (task: PlaygroundTask) => void
}

export interface ActivePlaygroundInvocationDraft {
  conversationId: string
  prepared: PreparedPlaygroundInvocation
}

export const usePlaygroundInvocation = ({
  invocation,
  isRoutingModelReady,
  onInvocationConsumed,
  routingModels,
  activateConversation,
  focusComposer,
  setConversationError,
  setDraft,
  setInputValue,
  setModel,
  startTask,
}: UsePlaygroundInvocationOptions) => {
  const consumedKeyRef = useRef<string | null>(null)

  useEffect(() => {
    if (!invocation || !isRoutingModelReady) return

    let prepared: PreparedPlaygroundInvocation
    try {
      prepared = preparePlaygroundInvocation(invocation)
    } catch {
      const conversationId = generateConversationId()
      activateConversation(conversationId, [])
      setDraft(null)
      setInputValue('')
      setConversationError(
        conversationId,
        'The Recipe run plan contains an unverified image and was blocked.',
      )
      onInvocationConsumed?.()
      return
    }
    if (consumedKeyRef.current === prepared.key) return
    consumedKeyRef.current = prepared.key

    const conversationId = generateConversationId()
    activateConversation(conversationId, toPlaygroundHistory(prepared.history, generateMessageId))

    const selectedModel = prepared.model
    if (!selectedModel || !routingModels.some((candidate) => candidate.id === selectedModel)) {
      setDraft(null)
      setInputValue('')
      setConversationError(
        conversationId,
        prepared.model
          ? `Probe model "${prepared.model}" is not advertised by the active router configuration.`
          : 'The Recipe run plan does not declare a request-facing model.',
      )
      onInvocationConsumed?.()
      return
    }

    setModel(selectedModel)
    setConversationError(conversationId, null)

    if (invocation.intent === 'edit') {
      if (!prepared.editable) {
        setDraft(null)
        setInputValue('')
        setConversationError(
          conversationId,
          'This structured probe does not end in a user message and cannot be edited without changing its semantics.',
        )
        onInvocationConsumed?.()
        return
      }
      setDraft({ conversationId, prepared })
      setInputValue(prepared.prompt)
      focusComposer()
    } else {
      setDraft(null)
      setInputValue('')
      startTask(createProbePlaygroundTask(prepared, conversationId, selectedModel))
    }

    onInvocationConsumed?.()
  }, [
    activateConversation,
    focusComposer,
    invocation,
    isRoutingModelReady,
    onInvocationConsumed,
    routingModels,
    setConversationError,
    setDraft,
    setInputValue,
    setModel,
    startTask,
  ])
}
