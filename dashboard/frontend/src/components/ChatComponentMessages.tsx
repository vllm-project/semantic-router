import styles from './ChatComponent.module.css'
import HeaderDisplay from './HeaderDisplay'
import ThinkingBlock from './ThinkingBlock'
import ErrorBoundary from './ErrorBoundary'
import ReMoMResponsesDisplay from './ReMoMResponsesDisplay'
import FeedbackButtons from './FeedbackButtons'
import { MessageActionBar, TypingGreeting } from './ChatComponentControls'
import { ContentWithCitations } from './ChatComponentCitations'
import { ToolCard } from './ChatComponentToolCards'
import { GREETING_LINES, type Message } from './ChatComponentTypes'
import { getTranslateAttr } from '../hooks/useNoTranslate'

interface ChatComponentMessagesProps {
  expandedToolCards: Set<string>
  messages: Message[]
  onToggleToolCard: (toolCallId: string) => void
}

interface ToolCallsProps {
  expandedToolCards: Set<string>
  message: Message
  onToggleToolCard: (toolCallId: string) => void
  wrapInBoundary?: boolean
}

function getSearchSources(message: Message) {
  return message.toolResults?.find(result => result.name === 'search_web')?.content
}

function ToolCalls({
  expandedToolCards,
  message,
  onToggleToolCard,
  wrapInBoundary = false,
}: ToolCallsProps) {
  if (!message.toolCalls?.length) {
    return null
  }

  return (
    <div className={styles.toolCallsContainer}>
      {message.toolCalls.map(toolCall => {
        const card = (
          <ToolCard
            key={toolCall.id}
            toolCall={toolCall}
            toolResult={message.toolResults?.find(result => result.callId === toolCall.id)}
            isExpanded={expandedToolCards.has(toolCall.id)}
            onToggle={() => onToggleToolCard(toolCall.id)}
          />
        )

        if (!wrapInBoundary) {
          return card
        }

        return <ErrorBoundary key={toolCall.id}>{card}</ErrorBoundary>
      })}
    </div>
  )
}

interface AssistantRatingsMessageProps {
  expandedToolCards: Set<string>
  message: Message
  onToggleToolCard: (toolCallId: string) => void
  prevUserQuery?: string
}

function AssistantRatingsMessage({
  expandedToolCards,
  message,
  onToggleToolCard,
  prevUserQuery,
}: AssistantRatingsMessageProps) {
  const otherModelIds = message.choices?.map(choice => choice.model).filter((model): model is string => model != null) ?? []
  const searchSources = getSearchSources(message)

  return (
    <>
      <ToolCalls
        expandedToolCards={expandedToolCards}
        message={message}
        onToggleToolCard={onToggleToolCard}
      />
      {message.thinkingProcess ? (
        <ThinkingBlock content={message.thinkingProcess} isStreaming={message.isStreaming} />
      ) : null}
      <div className={styles.ratingsChoices}>
        {message.choices?.map((choice, index) => (
          <div key={`${message.id}-${index}`} className={styles.choiceCard}>
            <div className={styles.choiceHeader}>
              <span className={styles.choiceModel}>{choice.model || `Model ${index + 1}`}</span>
              <span className={styles.choiceIndex}>Choice {index + 1}</span>
            </div>
            <div className={styles.choiceContent}>
              <ErrorBoundary>
                <ContentWithCitations
                  content={choice.content}
                  sources={searchSources}
                  isStreaming={message.isStreaming}
                />
              </ErrorBoundary>
              {message.isStreaming && index === 0 ? <span className={styles.cursor}>▊</span> : null}
            </div>
            {!message.isStreaming && choice.model ? (
              <div className={styles.choiceActions}>
                <FeedbackButtons
                  modelId={choice.model}
                  category={message.headers?.['x-vsr-selected-decision']}
                  query={prevUserQuery}
                  otherModelIds={otherModelIds.filter(model => model !== choice.model)}
                />
              </div>
            ) : null}
          </div>
        ))}
      </div>
    </>
  )
}

interface AssistantSingleMessageProps {
  expandedToolCards: Set<string>
  message: Message
  onToggleToolCard: (toolCallId: string) => void
}

function AssistantSingleMessage({
  expandedToolCards,
  message,
  onToggleToolCard,
}: AssistantSingleMessageProps) {
  const searchSources = getSearchSources(message)

  return (
    <>
      <ToolCalls
        expandedToolCards={expandedToolCards}
        message={message}
        onToggleToolCard={onToggleToolCard}
        wrapInBoundary
      />
      {message.thinkingProcess ? (
        <ThinkingBlock content={message.thinkingProcess} isStreaming={message.isStreaming} />
      ) : null}
      <div className={styles.messageText}>
        {message.content ? (
          <>
            <ErrorBoundary>
              <ContentWithCitations
                content={message.content}
                sources={searchSources}
                isStreaming={message.isStreaming}
              />
            </ErrorBoundary>
            {message.isStreaming ? <span className={styles.cursor}>▊</span> : null}
          </>
        ) : message.isStreaming ? (
          <span className={styles.cursor}>▊</span>
        ) : null}
      </div>
    </>
  )
}

interface MessageCardProps {
  expandedToolCards: Set<string>
  message: Message
  onToggleToolCard: (toolCallId: string) => void
  prevUserQuery?: string
}

function UserOrSystemMessage({ message }: Pick<MessageCardProps, 'message'>) {
  return (
    <div className={styles.messageText}>
      {message.content || message.isStreaming ? <span>{message.content}</span> : null}
      {message.isStreaming ? <span className={styles.cursor}>▊</span> : null}
    </div>
  )
}

function MessageCard({
  expandedToolCards,
  message,
  onToggleToolCard,
  prevUserQuery,
}: MessageCardProps) {
  const isRatingsMessage =
    message.role === 'assistant' && Boolean(message.choices && message.choices.length > 1)
  const showCopyAction =
    (message.role === 'assistant' || message.role === 'user') &&
    Boolean(message.content) &&
    !message.isStreaming

  return (
    <div
      className={`${styles.message} ${styles[message.role]}`}
      translate={getTranslateAttr(message.isStreaming ?? false)}
      data-message-id={message.id}
      data-message-role={message.role}
    >
      <div className={styles.messageContent} data-message-content>
        {message.role !== 'assistant' ? (
          <UserOrSystemMessage message={message} />
        ) : isRatingsMessage ? (
          <AssistantRatingsMessage
            expandedToolCards={expandedToolCards}
            message={message}
            onToggleToolCard={onToggleToolCard}
            prevUserQuery={prevUserQuery}
          />
        ) : (
          <AssistantSingleMessage
            expandedToolCards={expandedToolCards}
            message={message}
            onToggleToolCard={onToggleToolCard}
          />
        )}
        {message.role === 'assistant' && message.headers ? <HeaderDisplay headers={message.headers} /> : null}
        {message.role === 'assistant' && message.reasoning_mom_responses ? (
          <ReMoMResponsesDisplay rounds={message.reasoning_mom_responses} />
        ) : null}
        {showCopyAction ? (
          <div className={styles.messageActionRow}>
            <MessageActionBar content={message.content} />
            {message.role === 'assistant' && message.headers?.['x-vsr-selected-model'] ? (
              <FeedbackButtons
                modelId={message.headers['x-vsr-selected-model']}
                category={message.headers['x-vsr-selected-decision']}
                query={prevUserQuery}
              />
            ) : null}
          </div>
        ) : null}
      </div>
    </div>
  )
}

export default function ChatComponentMessages({
  expandedToolCards,
  messages,
  onToggleToolCard,
}: ChatComponentMessagesProps) {
  if (messages.length === 0) {
    return (
      <div className={`${styles.messagesContainer} ${styles.messagesContainerEmpty}`}>
        <div className={styles.emptyState}>
          <TypingGreeting lines={GREETING_LINES} />
        </div>
      </div>
    )
  }

  return (
    <div className={styles.messagesContainer}>
      <div className={styles.messages}>
        {messages.map((message, index) => {
          const prevUserQuery = messages[index - 1]?.role === 'user' ? messages[index - 1].content : undefined

          return (
            <MessageCard
              key={message.id}
              expandedToolCards={expandedToolCards}
              message={message}
              onToggleToolCard={onToggleToolCard}
              prevUserQuery={prevUserQuery}
            />
          )
        })}
      </div>
    </div>
  )
}
