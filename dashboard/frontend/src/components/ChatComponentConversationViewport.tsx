import styles from './ChatComponent.module.css'
import ChatComponentMessages from './ChatComponentMessages'
import type { Message } from './ChatComponentTypes'
import { useChatTranscriptAutoScroll } from './useChatTranscriptAutoScroll'

interface ChatComponentConversationViewportProps {
  canSubmitFeedback: boolean
  conversationId: string
  expandedToolCards: Set<string>
  messages: Message[]
  feedbackInsightsBasePath?: string
  onToggleToolCard: (toolCallId: string) => void
  thinking?: boolean
  thinkingProcess?: string
}

export default function ChatComponentConversationViewport({
  canSubmitFeedback,
  conversationId,
  expandedToolCards,
  messages,
  feedbackInsightsBasePath,
  onToggleToolCard,
  thinking = false,
  thinkingProcess,
}: ChatComponentConversationViewportProps) {
  const { containerRef, contentRef } = useChatTranscriptAutoScroll(messages, conversationId)

  return (
    <div className={styles.conversationViewport} ref={containerRef} data-testid="chat-transcript">
      <div className={styles.conversationViewportContent} ref={contentRef}>
        <ChatComponentMessages
          canSubmitFeedback={canSubmitFeedback}
          expandedToolCards={expandedToolCards}
          feedbackInsightsBasePath={feedbackInsightsBasePath}
          messages={messages}
          onToggleToolCard={onToggleToolCard}
          thinking={thinking}
          thinkingProcess={thinkingProcess}
        />
      </div>
    </div>
  )
}
