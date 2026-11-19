import {
  useCallback,
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
  type ChangeEvent,
  type KeyboardEventHandler,
  type Ref,
} from 'react'
import { ArrowUp, Mic, Square, X } from 'lucide-react'

import { Kbd, cn } from '../ui'
import ChatComposerAddMenu from './ChatComposerAddMenu'
import { useSpeechDictation } from '../hooks/useSpeechDictation'
import {
  formatPlaygroundFileSize,
  isPlaygroundImageAttachment,
  type PlaygroundAttachment,
} from './playgroundFileAttachments'

interface ChatComponentInputBarProps {
  attachments: PlaygroundAttachment[]
  attachFilesDisabled: boolean
  enableWebSearch: boolean
  inputRef: Ref<HTMLTextAreaElement>
  inputValue: string
  isLoading: boolean
  voiceInputDisabled: boolean
  webSearchDisabled?: boolean
  onAttachFiles: (files: FileList | File[]) => void
  onChangeInput: (value: string) => void
  onKeyDown: KeyboardEventHandler<HTMLTextAreaElement>
  onRemoveAttachment: (attachmentId: string) => void
  onSend: () => void
  onStop: () => void
  onToggleWebSearch: () => void
  sendDisabled?: boolean
  sendDisabledReason?: string
}

const iconButton =
  'grid size-8 shrink-0 place-items-center rounded-full outline-none transition-[color,background-color,transform] duration-150 focus-visible:ring-[3px] focus-visible:ring-ring/40 active:scale-95 disabled:pointer-events-none disabled:opacity-40 [&_svg]:size-4'

export default function ChatComponentInputBar({
  attachments,
  attachFilesDisabled,
  enableWebSearch,
  inputRef,
  inputValue,
  isLoading,
  voiceInputDisabled,
  webSearchDisabled = false,
  onAttachFiles,
  onChangeInput,
  onKeyDown,
  onRemoveAttachment,
  onSend,
  onStop,
  onToggleWebSearch,
  sendDisabled = false,
  sendDisabledReason,
}: ChatComponentInputBarProps) {
  const fileInputRef = useRef<HTMLInputElement>(null)
  const textareaRef = useRef<HTMLTextAreaElement | null>(null)
  const canSend = Boolean(inputValue.trim()) || attachments.length > 0
  const [isComposing, setIsComposing] = useState(false)
  const {
    isSupported: voiceSupported,
    isListening,
    toggleListening,
    stopListening,
  } = useSpeechDictation(onChangeInput)

  useEffect(() => {
    if (voiceInputDisabled && isListening) {
      stopListening()
    }
  }, [isListening, stopListening, voiceInputDisabled])

  useLayoutEffect(() => {
    const textarea = textareaRef.current
    if (!textarea) return

    textarea.style.height = 'auto'
    const maxHeight = 220
    const nextHeight = Math.min(textarea.scrollHeight, maxHeight)
    textarea.style.height = `${Math.max(nextHeight, 36)}px`
    textarea.style.overflowY = textarea.scrollHeight > maxHeight ? 'auto' : 'hidden'
  }, [inputValue])

  const setTextareaRef = useCallback(
    (node: HTMLTextAreaElement | null) => {
      textareaRef.current = node
      if (typeof inputRef === 'function') inputRef(node)
      else if (inputRef) (inputRef as { current: HTMLTextAreaElement | null }).current = node
    },
    [inputRef],
  )

  const handleChangeInput = useCallback(
    (value: string) => {
      if (isListening) {
        stopListening()
      }
      onChangeInput(value)
    },
    [isListening, onChangeInput, stopListening],
  )

  const handleKeyDown = useCallback<KeyboardEventHandler<HTMLTextAreaElement>>(
    (event) => {
      const nativeEvent = event.nativeEvent as KeyboardEvent & {
        isComposing?: boolean
        keyCode?: number
      }

      if (nativeEvent.isComposing || isComposing || nativeEvent.keyCode === 229) {
        return
      }

      onKeyDown(event)
    },
    [isComposing, onKeyDown],
  )

  const handleFileInputChange = useCallback(
    (event: ChangeEvent<HTMLInputElement>) => {
      const { files } = event.target
      if (files && files.length > 0) {
        onAttachFiles(files)
      }
      event.target.value = ''
    },
    [onAttachFiles],
  )

  const handleAttachClick = useCallback(() => {
    fileInputRef.current?.click()
  }, [])

  return (
    <div className="shrink-0 px-3 pb-3 pt-2 sm:px-4 sm:pb-4" data-testid="chat-composer">
      <div
        className={cn(
          'mx-auto w-full max-w-3xl rounded-2xl border border-border bg-card p-2 shadow-md transition-[border-color,box-shadow] duration-200',
          'focus-within:border-ring focus-within:shadow-[0_0_0_3px_var(--ring),var(--shadow-md)]',
        )}
      >
        {attachments.length > 0 ? (
          <div
            className="flex flex-wrap gap-2 px-1.5 pb-2 pt-1"
            data-testid="playground-attachment-list"
          >
            {attachments.map((attachment) =>
              isPlaygroundImageAttachment(attachment) ? (
                <figure
                  key={attachment.id}
                  className="relative flex w-28 flex-col overflow-hidden rounded-lg border border-border bg-surface-subtle"
                  data-testid={`playground-image-preview-${attachment.id}`}
                >
                  <img
                    src={attachment.content}
                    alt={`Preview of ${attachment.fileName}`}
                    className="h-20 w-full object-cover"
                  />
                  <figcaption
                    title={attachment.fileName}
                    className="flex flex-col px-2 py-1 text-[11px] leading-tight"
                  >
                    <span className="truncate font-medium">{attachment.fileName}</span>
                    <small className="text-subtle-foreground tabular">
                      {formatPlaygroundFileSize(attachment.sizeBytes)}
                    </small>
                  </figcaption>
                  <button
                    type="button"
                    className="absolute right-1 top-1 grid size-5 place-items-center rounded-full bg-background/80 text-foreground shadow-sm outline-none backdrop-blur-sm hover:bg-background focus-visible:ring-[3px] focus-visible:ring-ring/40"
                    onClick={() => onRemoveAttachment(attachment.id)}
                    aria-label={`Remove image ${attachment.fileName}`}
                    data-testid={`playground-attachment-remove-${attachment.id}`}
                  >
                    <X className="size-3" />
                  </button>
                </figure>
              ) : (
                <div
                  key={attachment.id}
                  className="inline-flex h-8 max-w-64 items-center gap-2 rounded-lg border border-border bg-surface-subtle pl-2.5 pr-1 text-xs"
                  data-testid={`playground-attachment-${attachment.id}`}
                >
                  <span className="truncate font-medium" title={attachment.fileName}>
                    {attachment.fileName}
                  </span>
                  <span className="shrink-0 text-subtle-foreground tabular">
                    {formatPlaygroundFileSize(attachment.sizeBytes)}
                  </span>
                  <button
                    type="button"
                    className="grid size-6 shrink-0 place-items-center rounded-md text-subtle-foreground outline-none hover:bg-accent hover:text-foreground focus-visible:ring-[3px] focus-visible:ring-ring/40"
                    onClick={() => onRemoveAttachment(attachment.id)}
                    aria-label={`Remove attachment ${attachment.fileName}`}
                    data-testid={`playground-attachment-remove-${attachment.id}`}
                  >
                    <X className="size-3.5" />
                  </button>
                </div>
              ),
            )}
          </div>
        ) : null}
        <textarea
          ref={setTextareaRef}
          value={inputValue}
          onChange={(event) => handleChangeInput(event.target.value)}
          onCompositionStart={() => setIsComposing(true)}
          onCompositionEnd={() => setIsComposing(false)}
          onKeyDown={handleKeyDown}
          placeholder="Ask me anything..."
          aria-label="Message"
          className="block w-full resize-none bg-transparent px-2.5 py-2 text-[14.5px] leading-relaxed text-foreground outline-none placeholder:text-subtle-foreground"
          rows={1}
        />
        <div className="flex items-center gap-1.5 px-1">
          <input
            ref={fileInputRef}
            type="file"
            multiple
            accept="text/*,.json,.md,.yaml,.yml,.csv,.tsv,image/gif,image/jpeg,image/png,image/webp"
            className="hidden"
            onChange={handleFileInputChange}
            aria-hidden="true"
            tabIndex={-1}
            data-testid="playground-attachment-input"
          />
          <ChatComposerAddMenu
            attachFilesDisabled={attachFilesDisabled || isLoading}
            onAttachFiles={handleAttachClick}
            onToggleWebSearch={onToggleWebSearch}
            webSearchDisabled={webSearchDisabled || isLoading}
            webSearchEnabled={enableWebSearch}
          />
          <span className="hidden items-center gap-1 text-xs text-subtle-foreground md:inline-flex">
            <Kbd>↵</Kbd> send <span aria-hidden>·</span> <Kbd>⇧</Kbd>
            <Kbd>↵</Kbd> new line
          </span>
          <div className="ml-auto flex items-center gap-1.5">
            {isLoading ? (
              <button
                type="button"
                className={cn(iconButton, 'bg-foreground text-background hover:bg-foreground/85')}
                onClick={onStop}
                title="Stop generating"
                aria-label="Stop generating"
              >
                <Square className="fill-current !size-3" />
              </button>
            ) : null}
            {voiceSupported ? (
              <button
                type="button"
                className={cn(
                  iconButton,
                  isListening
                    ? 'bg-destructive-soft text-destructive-foreground'
                    : 'text-muted-foreground hover:bg-accent hover:text-foreground',
                )}
                onClick={() => toggleListening(inputValue)}
                disabled={voiceInputDisabled}
                title={isListening ? 'Stop voice input' : 'Voice input'}
                aria-label={isListening ? 'Stop voice input' : 'Voice input'}
                aria-pressed={isListening}
              >
                <Mic />
              </button>
            ) : null}
            <button
              type="button"
              className={cn(
                iconButton,
                'bg-primary text-primary-foreground shadow-xs hover:bg-primary-hover',
              )}
              onClick={onSend}
              disabled={!canSend || sendDisabled}
              title={sendDisabled ? sendDisabledReason : 'Send message'}
              aria-label="Send message"
            >
              <ArrowUp strokeWidth={2.4} />
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}
