import type { Message } from './ChatComponentTypes'
import { isPlaygroundImageAttachment } from './playgroundFileAttachments'
import { containsInlineImageData } from '../hooks/persistenceSafety'

export const sanitizeMessagesForPersistence = (messages: Message[]): Message[] =>
  messages.map((message) => {
    const requestContent = containsInlineImageData(message.requestContent)
      ? undefined
      : message.requestContent
    const images = message.images?.filter((image) => !containsInlineImageData(image.src))
    const playgroundAttachments = message.playgroundAttachments?.filter(
      (attachment) => !isPlaygroundImageAttachment(attachment),
    )
    return {
      ...message,
      playgroundAttachments:
        playgroundAttachments && playgroundAttachments.length > 0
          ? playgroundAttachments
          : undefined,
      requestContent,
      images: images && images.length > 0 ? images : undefined,
    }
  })
