import { memo, useId, useState } from 'react'
import { ThinkingOrb } from 'thinking-orbs'
import { Bot, Code2, Plane, Scale, Sparkles, TriangleAlert } from 'lucide-react'

import { IconTile, LogoMark, ProviderLogo, cn } from '../ui'
import PlaygroundRouteLine from './PlaygroundRouteLine'
import ThinkingAnimation from './ThinkingAnimation'
import ThinkingBlock from './ThinkingBlock'
import ErrorBoundary from './ErrorBoundary'
import ReMoMResponsesDisplay from './ReMoMResponsesDisplay'
import FeedbackButtons from './FeedbackButtons'
import { MessageActionBar } from './ChatComponentControls'
import { ContentWithCitations } from './ChatComponentCitations'
import { ToolCard } from './ChatComponentToolCards'
import type { Message } from './ChatComponentTypes'
import { formatPlaygroundFileSize } from './playgroundFileAttachments'
import { getTranslateAttr } from '../hooks/useNoTranslate'
import { useAuth } from '../contexts/AuthContext'
import {
  X_VSR_REPLAY_ID,
  X_VSR_SELECTED_DECISION,
  X_VSR_SELECTED_MODEL,
} from '../generated/vsrHeaders'
import { modelProviderId, useInsightsConfig } from '../pages/insightsConfigFacts'
import { buildFeedbackInsightsHref } from './chatComponentSupport'

interface ChatComponentMessagesProps {
  canSubmitFeedback: boolean
  expandedToolCards: Set<string>
  feedbackInsightsBasePath?: string
  /** The answer the routing trace shows, which the transcript marks. */
  inspectedMessageId?: string
  messages: Message[]
  onInspectMessage?: (messageId: string) => void
  onSuggestion?: (prompt: string) => void
  onToggleToolCard: (toolCallId: string) => void
  thinking?: boolean
  thinkingProcess?: string
}

interface ToolCallsProps {
  expandedToolCards: Set<string>
  message: Message
  onToggleToolCard: (toolCallId: string) => void
  wrapInBoundary?: boolean
}

const SUGGESTIONS = [
  { icon: Scale, prompt: 'Draft a limitation-of-liability clause for a SaaS agreement' },
  { icon: Code2, prompt: 'Refactor this Python class to use dataclasses' },
  { icon: Bot, prompt: 'Why does my React effect run twice in development?' },
  { icon: Plane, prompt: 'Plan a three-day vegetarian trip to Kyoto' },
]

function StreamingResponseIndicator() {
  return (
    <span
      className="ml-1 inline-block size-4 align-[-0.26rem]"
      role="status"
      aria-label="Generating response"
    >
      <ThinkingOrb state="composing" size={20} />
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
    <div
      className="mb-3 grid grid-cols-[repeat(auto-fit,minmax(min(220px,100%),1fr))] gap-3"
      aria-label="Probe images"
    >
      {images.map((image, index) => (
        <img
          key={`${image.src.slice(0, 80)}-${index}`}
          className="block max-h-[32rem] w-full max-w-[40rem] rounded-lg border border-border bg-surface-sunken object-contain"
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
    <div className="flex w-full flex-col gap-2">
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
  canSubmitFeedback: boolean
  expandedToolCards: Set<string>
  feedbackInsightsBasePath?: string
  message: Message
  onToggleToolCard: (toolCallId: string) => void
  prevUserQuery?: string
}

function AssistantRatingsMessage({
  canSubmitFeedback,
  expandedToolCards,
  feedbackInsightsBasePath,
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
      <div className="grid w-full grid-cols-1 gap-3 lg:grid-cols-2">
        {message.choices?.map((choice, index) => (
          <div
            key={`${message.id}-${index}`}
            className="flex min-w-0 flex-col overflow-hidden rounded-xl border border-border bg-card shadow-xs"
          >
            <div className="flex items-center justify-between gap-2 border-b border-border bg-surface-subtle/70 px-3.5 py-2">
              <span className="truncate font-mono text-xs font-medium">
                {choice.model || `Model ${index + 1}`}
              </span>
              <span className="shrink-0 text-[11px] text-subtle-foreground">
                Choice {index + 1}
              </span>
            </div>
            <div className="min-w-0 px-3.5 py-3 text-[14px] leading-relaxed">
              <ErrorBoundary>
                <ContentWithCitations
                  content={choice.content}
                  sources={searchSources}
                  isStreaming={message.isStreaming}
                />
              </ErrorBoundary>
              {message.isStreaming && index === 0 ? <StreamingResponseIndicator /> : null}
            </div>
            {canSubmitFeedback &&
            !message.isStreaming &&
            !message.incomplete &&
            choice.model &&
            choice.model === message.headers?.[X_VSR_SELECTED_MODEL] &&
            message.headers?.[X_VSR_REPLAY_ID] ? (
              <div className="mt-auto border-t border-border px-2.5 py-1.5">
                <FeedbackButtons
                  modelId={choice.model}
                  replayId={message.headers[X_VSR_REPLAY_ID]}
                  insightsHref={buildFeedbackInsightsHref(
                    feedbackInsightsBasePath,
                    message.headers[X_VSR_REPLAY_ID],
                  )}
                  category={message.headers?.[X_VSR_SELECTED_DECISION]}
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
      <div className="w-full min-w-0 break-words text-[14.5px] leading-[1.7] text-foreground">
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
  canSubmitFeedback: boolean
  expandedToolCards: Set<string>
  feedbackInsightsBasePath?: string
  inspected: boolean
  message: Message
  onInspectMessage?: (messageId: string) => void
  onToggleToolCard: (toolCallId: string) => void
  prevUserQuery?: string
  /** What the turn waits for while its answer has nothing to show yet. */
  waitingLabel?: string
}

function WaitingForAnswer({ label }: { label: string }) {
  return (
    <span
      className="flex h-7 items-center gap-2 text-[13px] text-muted-foreground"
      role="status"
      aria-live="polite"
      data-testid="chat-thinking"
    >
      <ThinkingOrb state="composing" size={20} aria-label={label} />
      <span>{label}</span>
    </span>
  )
}

function UserMessageText({ content }: { content: string }) {
  const [expanded, setExpanded] = useState(false)
  const contentId = useId()
  // Bound the preview by both characters and lines without splitting a surrogate pair.
  const preview = content
    .slice(0, 1200)
    .replace(/[\uD800-\uDBFF]$/u, '')
    .split('\n', 9)
    .slice(0, 8)
    .join('\n')
  if (preview.length === content.length) return <span>{content}</span>

  return (
    <>
      <span id={contentId}>{expanded ? content : `${preview}…`}</span>
      <button
        type="button"
        className="mt-2 block rounded-sm text-xs font-medium text-primary outline-none hover:underline focus-visible:ring-[3px] focus-visible:ring-ring/40"
        aria-expanded={expanded}
        aria-controls={contentId}
        onClick={() => setExpanded((value) => !value)}
      >
        {expanded ? 'Show less' : 'Show more'}
      </button>
    </>
  )
}

function UserOrSystemMessage({ message }: Pick<MessageCardProps, 'message'>) {
  const attachmentItems = message.attachments ?? []
  const isUser = message.role === 'user'

  return (
    <div
      className={cn(
        'min-w-0 max-w-full break-words text-[14.5px] leading-relaxed',
        isUser
          ? 'whitespace-pre-wrap rounded-2xl rounded-br-md bg-secondary px-4 py-2.5 text-secondary-foreground'
          : 'w-full rounded-lg border border-dashed border-border-strong px-3.5 py-2.5 text-[13px] text-muted-foreground',
      )}
    >
      {attachmentItems.length > 0 ? (
        <div className="mb-2 flex flex-wrap justify-end gap-1.5">
          {attachmentItems.map((attachment) => (
            <span
              key={`${attachment.fileName}-${attachment.sizeBytes}`}
              className="inline-flex max-w-56 items-center gap-1.5 rounded-md border border-border bg-surface px-2 py-0.5 text-xs"
              title={attachment.fileName}
            >
              <span className="truncate">{attachment.fileName}</span>
              <span className="shrink-0 text-subtle-foreground tabular">
                {formatPlaygroundFileSize(attachment.sizeBytes)}
              </span>
            </span>
          ))}
        </div>
      ) : null}
      <MessageImages message={message} />
      {message.content || message.isStreaming ? (
        isUser ? (
          <UserMessageText content={message.content} />
        ) : (
          <span>{message.content}</span>
        )
      ) : null}
      {message.isStreaming ? <StreamingResponseIndicator /> : null}
    </div>
  )
}

function AssistantAvatar({ message }: { message: Message }) {
  const { modelProviders } = useInsightsConfig()
  const model = message.headers?.[X_VSR_SELECTED_MODEL]
  const provider = model ? modelProviderId(model, modelProviders) : undefined
  return (
    <span className="mt-0.5 grid size-7 shrink-0 place-items-center rounded-full border border-border bg-surface shadow-xs">
      {provider ? <ProviderLogo provider={provider} size={14} /> : <LogoMark size={14} />}
    </span>
  )
}

const MessageCard = memo(
  function MessageCard({
    canSubmitFeedback,
    expandedToolCards,
    feedbackInsightsBasePath,
    inspected,
    message,
    onInspectMessage,
    onToggleToolCard,
    prevUserQuery,
    waitingLabel,
  }: MessageCardProps) {
    const isAssistant = message.role === 'assistant'
    const isRatingsMessage = isAssistant && Boolean(message.choices && message.choices.length > 1)
    const showCopyAction =
      (isAssistant || message.role === 'user') &&
      (Boolean(message.content) ||
        (message.attachments?.length ?? 0) > 0 ||
        (message.images?.length ?? 0) > 0) &&
      !message.isStreaming
    const hasRoute = isAssistant && Boolean(message.headers)

    return (
      <div
        className={cn('group/message flex min-w-0 gap-3', !isAssistant && 'justify-end')}
        translate={getTranslateAttr(message.isStreaming ?? false)}
        data-message-id={message.id}
        data-message-role={message.role}
      >
        {isAssistant ? <AssistantAvatar message={message} /> : null}
        <div
          className={cn(
            'flex min-w-0 flex-col gap-3',
            isAssistant ? 'flex-1 items-start' : 'max-w-[min(80%,560px)] items-end',
          )}
          data-message-content
        >
          {!isAssistant ? (
            <UserOrSystemMessage message={message} />
          ) : waitingLabel &&
            !message.content &&
            !message.toolCalls?.length &&
            !message.thinkingProcess ? (
            <WaitingForAnswer label={waitingLabel} />
          ) : isRatingsMessage ? (
            <AssistantRatingsMessage
              canSubmitFeedback={canSubmitFeedback}
              expandedToolCards={expandedToolCards}
              feedbackInsightsBasePath={feedbackInsightsBasePath}
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
          {isAssistant && message.incomplete ? (
            <div
              className="flex w-full items-start gap-2 rounded-lg border border-warning/30 bg-warning-soft px-3 py-2 text-[13px]"
              role="status"
            >
              <TriangleAlert className="mt-0.5 size-4 shrink-0 text-warning" aria-hidden />
              <span>
                <strong className="font-medium">Incomplete response.</strong>{' '}
                <span className="text-muted-foreground">{message.incomplete}</span>
              </span>
            </div>
          ) : null}
          {isAssistant && message.reasoning_mom_responses ? (
            <ReMoMResponsesDisplay rounds={message.reasoning_mom_responses} />
          ) : null}
          {showCopyAction || hasRoute ? (
            <div
              className={cn(
                'flex min-w-0 max-w-full flex-wrap items-center gap-x-2 gap-y-1.5',
                !isAssistant &&
                  'opacity-0 transition-opacity focus-within:opacity-100 group-hover/message:opacity-100',
              )}
            >
              {hasRoute && message.headers ? (
                <PlaygroundRouteLine
                  headers={message.headers}
                  inspected={inspected}
                  onInspect={onInspectMessage ? () => onInspectMessage(message.id) : undefined}
                />
              ) : null}
              <div className="flex items-center gap-0.5">
                {showCopyAction ? <MessageActionBar content={message.content} /> : null}
                {canSubmitFeedback &&
                isAssistant &&
                !message.incomplete &&
                message.headers?.[X_VSR_SELECTED_MODEL] &&
                message.headers?.[X_VSR_REPLAY_ID] ? (
                  <FeedbackButtons
                    modelId={message.headers[X_VSR_SELECTED_MODEL]}
                    replayId={message.headers[X_VSR_REPLAY_ID]}
                    insightsHref={buildFeedbackInsightsHref(
                      feedbackInsightsBasePath,
                      message.headers[X_VSR_REPLAY_ID],
                    )}
                    category={message.headers[X_VSR_SELECTED_DECISION]}
                    query={prevUserQuery}
                  />
                ) : null}
              </div>
            </div>
          ) : null}
        </div>
      </div>
    )
  },
  (prevProps, nextProps) =>
    prevProps.canSubmitFeedback === nextProps.canSubmitFeedback &&
    prevProps.feedbackInsightsBasePath === nextProps.feedbackInsightsBasePath &&
    prevProps.inspected === nextProps.inspected &&
    prevProps.message === nextProps.message &&
    prevProps.prevUserQuery === nextProps.prevUserQuery &&
    prevProps.waitingLabel === nextProps.waitingLabel &&
    prevProps.onInspectMessage === nextProps.onInspectMessage &&
    prevProps.onToggleToolCard === nextProps.onToggleToolCard &&
    prevProps.expandedToolCards === nextProps.expandedToolCards,
)

function EmptyConversation({ onSuggestion }: { onSuggestion?: (prompt: string) => void }) {
  const { user } = useAuth()
  const firstName = user?.name?.trim().split(/\s+/)[0] || 'there'
  return (
    <div className="mx-auto flex w-full max-w-2xl flex-1 flex-col items-center justify-center px-6 pb-10 pt-12 text-center motion-safe:animate-fade-up">
      <IconTile icon={Sparkles} tone="blue" size="xl" className="mb-5" />
      <h2 className="text-2xl font-semibold tracking-[-0.025em]">Welcome, {firstName}</h2>
      <p className="mt-2 max-w-md text-[13.5px] leading-relaxed text-muted-foreground">
        Every answer shows which signals fired, the decision that matched and the model that
        answered.
      </p>
      {onSuggestion ? (
        <div className="mt-8 grid w-full grid-cols-1 gap-2 sm:grid-cols-2">
          {SUGGESTIONS.map((suggestion) => (
            <button
              key={suggestion.prompt}
              type="button"
              onClick={() => onSuggestion(suggestion.prompt)}
              className="group flex items-start gap-3 rounded-xl border border-border bg-card p-3 text-left shadow-xs outline-none transition-[border-color,box-shadow,transform] duration-200 hover:-translate-y-px hover:border-border-strong hover:shadow-md focus-visible:ring-[3px] focus-visible:ring-ring/40"
            >
              <suggestion.icon className="mt-0.5 size-4 shrink-0 text-subtle-foreground transition-colors group-hover:text-primary" />
              <span className="text-[13px] font-medium leading-snug">{suggestion.prompt}</span>
            </button>
          ))}
        </div>
      ) : null}
    </div>
  )
}

export default function ChatComponentMessages({
  canSubmitFeedback,
  expandedToolCards,
  feedbackInsightsBasePath,
  inspectedMessageId,
  messages,
  onInspectMessage,
  onSuggestion,
  onToggleToolCard,
  thinking = false,
  thinkingProcess,
}: ChatComponentMessagesProps) {
  if (messages.length === 0 && !thinking) {
    return <EmptyConversation onSuggestion={onSuggestion} />
  }

  // The turn's own answer row shows the wait, so the rail never shows a second avatar.
  const last = messages[messages.length - 1]
  const answerPending = thinking && last?.role === 'assistant' && Boolean(last.isStreaming)
  const waitingLabel = thinkingProcess ? 'Working' : 'Routing your message'

  return (
    <div
      className="mx-auto flex w-full max-w-3xl flex-col gap-7 px-4 pb-8 pt-6 sm:px-6"
      data-testid="chat-message-rail"
    >
      {messages.map((message, index) => {
        const prevUserQuery =
          messages[index - 1]?.role === 'user' ? messages[index - 1].content : undefined

        return (
          <MessageCard
            key={message.id}
            canSubmitFeedback={canSubmitFeedback}
            expandedToolCards={expandedToolCards}
            feedbackInsightsBasePath={feedbackInsightsBasePath}
            inspected={message.id === inspectedMessageId}
            message={message}
            onInspectMessage={onInspectMessage}
            onToggleToolCard={onToggleToolCard}
            prevUserQuery={prevUserQuery}
            waitingLabel={answerPending && message === last ? waitingLabel : undefined}
          />
        )
      })}
      {thinking && !answerPending ? <ThinkingAnimation thinkingProcess={thinkingProcess} /> : null}
    </div>
  )
}
