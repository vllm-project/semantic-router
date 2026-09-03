import { useCallback, type MutableRefObject } from 'react'

import {
  generatePlaygroundTaskId,
  type Message,
  type PlaygroundTask,
  type PlaygroundTaskRequestOptions,
} from './ChatComponentTypes'
import { materializeEditedProbeRequest } from './playgroundInvocationSupport'
import type { ActivePlaygroundInvocationDraft } from './usePlaygroundInvocation'
import type { PlaygroundAttachment } from './playgroundFileAttachments'

interface UsePlaygroundTaskSubmissionOptions {
  activeTasksRef: MutableRefObject<Record<string, PlaygroundTask>>
  buildTaskRequestOptions: () => PlaygroundTaskRequestOptions
  clearPendingAttachments: () => void
  conversationId: string
  conversations: Array<{ id: string }>
  copyPendingAttachmentsForTask: () => PlaygroundAttachment[]
  enqueueTask: (task: PlaygroundTask) => void
  getConversationMessagesSnapshot: (conversationId: string) => Message[]
  hasHydratedConversation: MutableRefObject<boolean>
  inputValue: string
  isRoutingModelReady: boolean
  model: string
  pendingAttachments: PlaygroundAttachment[]
  probeDraft: ActivePlaygroundInvocationDraft | null
  saveConversation: (conversationId: string, messages: Message[]) => void
  setConversationError: (conversationId: string, error: string | null) => void
  setInputValue: (value: string) => void
  setProbeDraft: (draft: ActivePlaygroundInvocationDraft | null) => void
  startTask: (task: PlaygroundTask) => void
}

export const usePlaygroundTaskSubmission = ({
  activeTasksRef,
  buildTaskRequestOptions,
  clearPendingAttachments,
  conversationId,
  conversations,
  copyPendingAttachmentsForTask,
  enqueueTask,
  getConversationMessagesSnapshot,
  hasHydratedConversation,
  inputValue,
  isRoutingModelReady,
  model,
  pendingAttachments,
  probeDraft,
  saveConversation,
  setConversationError,
  setInputValue,
  setProbeDraft,
  startTask,
}: UsePlaygroundTaskSubmissionOptions) =>
  useCallback(() => {
    if (!isRoutingModelReady) return
    const trimmedInput = inputValue.trim()
    if (!trimmedInput && pendingAttachments.length === 0) return

    const attachmentsForTask = copyPendingAttachmentsForTask()
    const activeProbeDraft =
      probeDraft?.conversationId === conversationId ? probeDraft.prepared : null
    const nextTask: PlaygroundTask = {
      id: generatePlaygroundTaskId(),
      conversationId,
      prompt: trimmedInput,
      attachments: attachmentsForTask.length > 0 ? attachmentsForTask : undefined,
      createdAt: Date.now(),
      requestOptions: activeProbeDraft
        ? {
            enableClawMode: false,
            enableWebSearch: false,
            executeToolCalls: false,
            model,
          }
        : buildTaskRequestOptions(),
      exactRequest: activeProbeDraft
        ? materializeEditedProbeRequest(activeProbeDraft, trimmedInput, model, attachmentsForTask)
        : undefined,
      displayMessage: activeProbeDraft
        ? {
            content: trimmedInput,
            ...(activeProbeDraft.promptImages?.length
              ? { images: activeProbeDraft.promptImages }
              : {}),
          }
        : undefined,
    }

    if (!conversations.some((conversation) => conversation.id === conversationId)) {
      hasHydratedConversation.current = true
      saveConversation(conversationId, getConversationMessagesSnapshot(conversationId))
    }

    setConversationError(conversationId, null)
    setInputValue('')
    setProbeDraft(null)
    clearPendingAttachments()

    if (!activeTasksRef.current[conversationId]) {
      startTask(nextTask)
      return
    }
    enqueueTask(nextTask)
  }, [
    activeTasksRef,
    buildTaskRequestOptions,
    clearPendingAttachments,
    conversationId,
    conversations,
    copyPendingAttachmentsForTask,
    enqueueTask,
    getConversationMessagesSnapshot,
    hasHydratedConversation,
    inputValue,
    isRoutingModelReady,
    model,
    pendingAttachments,
    probeDraft,
    saveConversation,
    setConversationError,
    setInputValue,
    setProbeDraft,
    startTask,
  ])
