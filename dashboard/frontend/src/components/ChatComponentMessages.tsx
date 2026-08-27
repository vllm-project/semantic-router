import { memo } from 'react'
import { ThinkingOrb } from 'thinking-orbs'

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
import { formatPlaygroundFileSize } from './playgroundFileAttachments'
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

function StreamingResponseIndicator() {
  return (
    <span className={styles.streamingIndicator} role="status" aria-label="Generating response">
      <ThinkingOrb state="composing" size={20} theme="dark" />
    </span>
  )
}

function getSearchSources(message: Message) {
  return message.toolResults?.find((result) => result.name === 'search_web')?.content
}

function MessageImages({ message }: { message: Message }) {
  const images = message.images ?? []
  if (images.length === 0) return null

  return (
    <div className={styles.messageImageGrid} aria-label="Probe images">
      {images.map((image, index) => (
        <img
          key={`${image.src.slice(0, 80)}-${index}`}
          className={styles.messageImage}
          src={image.src}
          alt={image.alt}
          data-testid="probe-message-image"
        />
      ))}
    </div>
  )
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
      {message.toolCalls.map((toolCall) => {
        const card = (
          <ToolCard
            key={toolCall.id}
            toolCall={toolCall}
            toolResult={message.toolResults?.find((result) => result.callId === toolCall.id)}
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
      <MessageImages message={message} />
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
              {message.isStreaming && index === 0 ? <StreamingResponseIndicator /> : null}
            </div>
            {!message.isStreaming && choice.model && message.headers?.['x-vsr-replay-id'] ? (
              <div className={styles.choiceActions}>
                <FeedbackButtons
                  modelId={choice.model}
                  replayId={message.headers['x-vsr-replay-id']}
                  category={message.headers?.['x-vsr-selected-decision']}
                  query={prevUserQuery}
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
        <MessageImages message={message} />
        {message.content ? (
          <>
            <ErrorBoundary>
              <ContentWithCitations
                content={message.content}
                sources={searchSources}
                isStreaming={message.isStreaming}
              />
            </ErrorBoundary>
            {message.isStreaming ? <StreamingResponseIndicator /> : null}
          </>
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
  const attachmentItems = message.attachments ?? []

  return (
    <div className={styles.messageText}>
      {attachmentItems.length > 0 ? (
        <div className={styles.messageAttachmentList}>
          {attachmentItems.map((attachment) => (
            <span
              key={`${attachment.fileName}-${attachment.sizeBytes}`}
              className={styles.messageAttachmentChip}
              title={attachment.fileName}
            >
              {attachment.fileName}
              <span className={styles.messageAttachmentChipSize}>
                {formatPlaygroundFileSize(attachment.sizeBytes)}
              </span>
            </span>
          ))}
        </div>
      ) : null}
      <MessageImages message={message} />
      {message.content || message.isStreaming ? <span>{message.content}</span> : null}
      {message.isStreaming ? <StreamingResponseIndicator /> : null}
    </div>
  )
}

const MessageCard = memo(
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
      (Boolean(message.content) ||
        (message.attachments?.length ?? 0) > 0 ||
        (message.images?.length ?? 0) > 0) &&
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
          {message.role === 'assistant' && message.headers ? (
            <HeaderDisplay headers={message.headers} />
          ) : null}
          {message.role === 'assistant' && message.reasoning_mom_responses ? (
            <ReMoMResponsesDisplay rounds={message.reasoning_mom_responses} />
          ) : null}
          {showCopyAction ? (
            <div className={styles.messageActionRow}>
              <MessageActionBar content={message.content} />
              {message.role === 'assistant' &&
              message.headers?.['x-vsr-selected-model'] &&
              message.headers?.['x-vsr-replay-id'] ? (
                <FeedbackButtons
                  modelId={message.headers['x-vsr-selected-model']}
                  replayId={message.headers['x-vsr-replay-id']}
                  category={message.headers['x-vsr-selected-decision']}
                  query={prevUserQuery}
                />
              ) : null}
            </div>
          ) : null}
        </div>
      </div>
    )
  },
  (prevProps, nextProps) =>
    prevProps.message === nextProps.message &&
    prevProps.prevUserQuery === nextProps.prevUserQuery &&
    prevProps.onToggleToolCard === nextProps.onToggleToolCard &&
    prevProps.expandedToolCards === nextProps.expandedToolCards,
)

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
          const prevUserQuery =
            messages[index - 1]?.role === 'user' ? messages[index - 1].content : undefined

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
