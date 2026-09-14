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

import styles from './ChatComponent.module.css'
import ChatComposerAddMenu from './ChatComposerAddMenu'
import ChatComposerModelSelect from './ChatComposerModelSelect'
import { useSpeechDictation } from '../hooks/useSpeechDictation'
import type { RouterModelOption } from '../utils/routerModelSelection'
import {
  formatPlaygroundFileSize,
  isPlaygroundImageAttachment,
  type PlaygroundAttachment,
} from './playgroundFileAttachments'

interface ChatComponentInputBarProps {
  attachments: PlaygroundAttachment[]
  attachFilesDisabled: boolean
  enableClawMode: boolean
  enableWebSearch: boolean
  inputRef: Ref<HTMLTextAreaElement>
  inputValue: string
  isLoading: boolean
  isTogglingClawMode: boolean
  modeToggleDisabled: boolean
  modelOptions: RouterModelOption[]
  modelSelectDisabled: boolean
  selectedModel: string
  voiceInputDisabled: boolean
  webSearchDisabled?: boolean
  onAttachFiles: (files: FileList | File[]) => void
  onChangeInput: (value: string) => void
  onKeyDown: KeyboardEventHandler<HTMLTextAreaElement>
  onModelChange: (model: string) => void
  onRemoveAttachment: (attachmentId: string) => void
  onSend: () => void
  onStop: () => void
  onToggleClawMode: () => void
  onToggleClawRoom: () => void
  onToggleWebSearch: () => void
  sendDisabled?: boolean
  sendDisabledReason?: string
  showClawRoom: boolean
}

export default function ChatComponentInputBar({
  attachments,
  attachFilesDisabled,
  enableClawMode,
  enableWebSearch,
  inputRef,
  inputValue,
  isLoading,
  isTogglingClawMode,
  modeToggleDisabled,
  modelOptions,
  modelSelectDisabled,
  selectedModel,
  voiceInputDisabled,
  webSearchDisabled = false,
  onAttachFiles,
  onChangeInput,
  onKeyDown,
  onModelChange,
  onRemoveAttachment,
  onSend,
  onStop,
  onToggleClawMode,
  onToggleClawRoom,
  onToggleWebSearch,
  sendDisabled = false,
  sendDisabledReason,
  showClawRoom,
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
    <div className={styles.inputContainer} data-testid="chat-composer">
      <div className={`${styles.inputWrapper} ${canSend ? styles.hasContent : ''}`}>
        {attachments.length > 0 ? (
          <div className={styles.attachmentList} data-testid="playground-attachment-list">
            {attachments.map((attachment) =>
              isPlaygroundImageAttachment(attachment) ? (
                <figure
                  key={attachment.id}
                  className={styles.attachmentImagePreview}
                  data-testid={`playground-image-preview-${attachment.id}`}
                >
                  <img src={attachment.content} alt={`Preview of ${attachment.fileName}`} />
                  <figcaption title={attachment.fileName}>
                    <span>{attachment.fileName}</span>
                    <small>{formatPlaygroundFileSize(attachment.sizeBytes)}</small>
                  </figcaption>
                  <button
                    type="button"
                    className={styles.attachmentImageRemove}
                    onClick={() => onRemoveAttachment(attachment.id)}
                    aria-label={`Remove image ${attachment.fileName}`}
                    data-testid={`playground-attachment-remove-${attachment.id}`}
                  >
                    ×
                  </button>
                </figure>
              ) : (
                <div
                  key={attachment.id}
                  className={styles.attachmentChip}
                  data-testid={`playground-attachment-${attachment.id}`}
                >
                  <span className={styles.attachmentChipName} title={attachment.fileName}>
                    {attachment.fileName}
                  </span>
                  <span className={styles.attachmentChipSize}>
                    {formatPlaygroundFileSize(attachment.sizeBytes)}
                  </span>
                  <button
                    type="button"
                    className={styles.attachmentChipRemove}
                    onClick={() => onRemoveAttachment(attachment.id)}
                    aria-label={`Remove attachment ${attachment.fileName}`}
                    data-testid={`playground-attachment-remove-${attachment.id}`}
                  >
                    ×
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
          className={styles.input}
          rows={1}
        />
        <div className={styles.inputActionsRow}>
          <div className={styles.inputActions}>
            <input
              ref={fileInputRef}
              type="file"
              multiple
              accept="text/*,.json,.md,.yaml,.yml,.csv,.tsv,image/gif,image/jpeg,image/png,image/webp"
              className={styles.attachmentFileInput}
              onChange={handleFileInputChange}
              aria-hidden="true"
              tabIndex={-1}
              data-testid="playground-attachment-input"
            />
            <ChatComposerAddMenu
              attachFilesDisabled={attachFilesDisabled || isLoading || isTogglingClawMode}
              clawModeDisabled={modeToggleDisabled}
              clawModeEnabled={enableClawMode}
              clawRoom={
                showClawRoom
                  ? {
                      active: false,
                      disabled: modeToggleDisabled,
                      onToggle: onToggleClawRoom,
                    }
                  : undefined
              }
              onAttachFiles={handleAttachClick}
              onToggleClawMode={onToggleClawMode}
              onToggleWebSearch={onToggleWebSearch}
              webSearchDisabled={webSearchDisabled || isLoading || isTogglingClawMode}
              webSearchEnabled={enableWebSearch}
            />
            <ChatComposerModelSelect
              disabled={modelSelectDisabled}
              models={modelOptions}
              onChange={onModelChange}
              value={selectedModel}
            />
          </div>
          <div className={styles.composerButtons}>
            {isLoading ? (
              <button
                className={`${styles.sendButton} ${styles.stopButton}`}
                onClick={onStop}
                title="Stop generating"
                aria-label="Stop generating"
              >
                <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
                  <rect x="6" y="6" width="12" height="12" rx="2" />
                </svg>
              </button>
            ) : null}
            {voiceSupported ? (
              <button
                type="button"
                className={`${styles.sendButton} ${isListening ? styles.sendButtonVoiceListening : ''}`}
                onClick={() => toggleListening(inputValue)}
                disabled={voiceInputDisabled}
                title={isListening ? 'Stop voice input' : 'Voice input'}
                aria-label={isListening ? 'Stop voice input' : 'Voice input'}
                aria-pressed={isListening}
              >
                <svg
                  width="16"
                  height="16"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2.5"
                >
                  <path
                    d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  />
                  <path
                    d="M19 10v2a7 7 0 0 1-14 0v-2"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  />
                  <line x1="12" y1="19" x2="12" y2="23" strokeLinecap="round" />
                  <line x1="8" y1="23" x2="16" y2="23" strokeLinecap="round" />
                </svg>
              </button>
            ) : null}
            <button
              className={styles.sendButton}
              onClick={onSend}
              disabled={!canSend || sendDisabled}
              title={sendDisabled ? sendDisabledReason : 'Send message'}
              aria-label="Send message"
            >
              <svg
                width="16"
                height="16"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2.5"
              >
                <path d="M12 19V5M5 12l7-7 7 7" strokeLinecap="round" strokeLinejoin="round" />
              </svg>
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}
