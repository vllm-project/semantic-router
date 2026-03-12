import type { ReactNode } from 'react'

import styles from './ChatComponent.module.css'
import ChatComponentMessages from './ChatComponentMessages'
import type { Message } from './ChatComponentTypes'
import { useChatTranscriptAutoScroll } from './useChatTranscriptAutoScroll'

interface ChatComponentConversationViewportProps {
  expandedToolCards: Set<string>
  footer: ReactNode
  messages: Message[]
  onToggleToolCard: (toolCallId: string) => void
}

export default function ChatComponentConversationViewport({
  expandedToolCards,
  footer,
  messages,
  onToggleToolCard,
}: ChatComponentConversationViewportProps) {
  const { containerRef, contentRef } = useChatTranscriptAutoScroll(messages)

  return (
    <div className={styles.conversationViewport} ref={containerRef} data-testid="chat-transcript">
      <div className={styles.conversationViewportContent} ref={contentRef}>
        <ChatComponentMessages
          expandedToolCards={expandedToolCards}
          messages={messages}
          onToggleToolCard={onToggleToolCard}
        />
        {footer}
      </div>
    </div>
  )
}
