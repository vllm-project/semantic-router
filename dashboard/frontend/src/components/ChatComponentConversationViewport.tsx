import ChatComponentMessages from './ChatComponentMessages'
import type { Message } from './ChatComponentTypes'
import { useChatTranscriptAutoScroll } from './useChatTranscriptAutoScroll'

interface ChatComponentConversationViewportProps {
  canSubmitFeedback: boolean
  conversationId: string
  expandedToolCards: Set<string>
  inspectedMessageId?: string
  messages: Message[]
  feedbackInsightsBasePath?: string
  onInspectMessage?: (messageId: string) => void
  onSuggestion?: (prompt: string) => void
  onToggleToolCard: (toolCallId: string) => void
  thinking?: boolean
  thinkingProcess?: string
}

export default function ChatComponentConversationViewport({
  canSubmitFeedback,
  conversationId,
  expandedToolCards,
  inspectedMessageId,
  messages,
  feedbackInsightsBasePath,
  onInspectMessage,
  onSuggestion,
  onToggleToolCard,
  thinking = false,
  thinkingProcess,
}: ChatComponentConversationViewportProps) {
  const { containerRef, contentRef } = useChatTranscriptAutoScroll(messages, conversationId)

  return (
    <div
      className="min-h-0 flex-1 overflow-y-auto overflow-x-hidden [scrollbar-gutter:stable]"
      ref={containerRef}
      data-testid="chat-transcript"
    >
      <div className="flex min-h-full flex-col" ref={contentRef}>
        <ChatComponentMessages
          canSubmitFeedback={canSubmitFeedback}
          expandedToolCards={expandedToolCards}
          feedbackInsightsBasePath={feedbackInsightsBasePath}
          inspectedMessageId={inspectedMessageId}
          messages={messages}
          onInspectMessage={onInspectMessage}
          onSuggestion={onSuggestion}
          onToggleToolCard={onToggleToolCard}
          thinking={thinking}
          thinkingProcess={thinkingProcess}
        />
      </div>
    </div>
  )
}
