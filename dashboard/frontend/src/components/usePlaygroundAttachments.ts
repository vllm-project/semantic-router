import { useCallback, useState } from 'react'

import {
  readPlaygroundAttachmentFile,
  type PlaygroundAttachment,
} from './playgroundFileAttachments'

interface UsePlaygroundAttachmentsOptions {
  conversationId: string
  setConversationError: (targetConversationId: string, error: string | null) => void
}

export const usePlaygroundAttachments = ({
  conversationId,
  setConversationError,
}: UsePlaygroundAttachmentsOptions) => {
  const [pendingAttachments, setPendingAttachments] = useState<PlaygroundAttachment[]>([])

  const handleAttachFiles = useCallback(async (files: FileList | File[]) => {
    const nextFiles = Array.from(files)
    if (nextFiles.length === 0) {
      return
    }

    setConversationError(conversationId, null)

    for (const file of nextFiles) {
      try {
        const attachment = await readPlaygroundAttachmentFile(file)
        setPendingAttachments(prev => [...prev, attachment])
      } catch (error) {
        const message = error instanceof Error ? error.message : `Could not attach "${file.name}".`
        setConversationError(conversationId, message)
        break
      }
    }
  }, [conversationId, setConversationError])

  const handleRemoveAttachment = useCallback((attachmentId: string) => {
    setPendingAttachments(prev => prev.filter(attachment => attachment.id !== attachmentId))
  }, [])

  const clearPendingAttachments = useCallback(() => {
    setPendingAttachments([])
  }, [])

  const restorePendingAttachments = useCallback((attachments: PlaygroundAttachment[] = []) => {
    setPendingAttachments(attachments.map(attachment => ({ ...attachment })))
  }, [])

  const copyPendingAttachmentsForTask = useCallback(
    () => pendingAttachments.map(attachment => ({ ...attachment })),
    [pendingAttachments]
  )

  return {
    clearPendingAttachments,
    copyPendingAttachmentsForTask,
    handleAttachFiles,
    handleRemoveAttachment,
    pendingAttachments,
    restorePendingAttachments,
  }
}
