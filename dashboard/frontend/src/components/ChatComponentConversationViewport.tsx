import styles from './ChatComponent.module.css'
import ChatComponentMessages from './ChatComponentMessages'
import type { Message } from './ChatComponentTypes'
import { useChatTranscriptAutoScroll } from './useChatTranscriptAutoScroll'

interface ChatComponentConversationViewportProps {
  conversationId: string
  expandedToolCards: Set<string>
  messages: Message[]
  onToggleToolCard: (toolCallId: string) => void
}

export default function ChatComponentConversationViewport({
  conversationId,
  expandedToolCards,
  messages,
  onToggleToolCard,
}: ChatComponentConversationViewportProps) {
  const { containerRef, contentRef } = useChatTranscriptAutoScroll(messages, conversationId)

  return (
    <div className={styles.conversationViewport} ref={containerRef} data-testid="chat-transcript">
      <div className={styles.conversationViewportContent} ref={contentRef}>
        <ChatComponentMessages
          expandedToolCards={expandedToolCards}
          messages={messages}
          onToggleToolCard={onToggleToolCard}
        />
      </div>
    </div>
  )
}
