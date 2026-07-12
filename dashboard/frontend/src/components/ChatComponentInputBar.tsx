import { useCallback, useEffect, useRef, useState, type ChangeEvent, type KeyboardEventHandler, type ReactNode, type Ref } from 'react'

import styles from './ChatComponent.module.css'
import { ClawModeToggle, ToolToggle } from './ChatComponentControls'
import { useSpeechDictation } from '../hooks/useSpeechDictation'
import {
  formatPlaygroundFileSize,
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
  voiceInputDisabled: boolean
  onAttachFiles: (files: FileList | File[]) => void
  onChangeInput: (value: string) => void
  onKeyDown: KeyboardEventHandler<HTMLTextAreaElement>
  onRemoveAttachment: (attachmentId: string) => void
  onSend: () => void
  onStop: () => void
  onToggleClawMode: () => void
  onToggleWebSearch: () => void
  roomChatToggleControl: ReactNode
  sendDisabled?: boolean
  sendDisabledReason?: string
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
  voiceInputDisabled,
  onAttachFiles,
  onChangeInput,
  onKeyDown,
  onRemoveAttachment,
  onSend,
  onStop,
  onToggleClawMode,
  onToggleWebSearch,
  roomChatToggleControl,
  sendDisabled = false,
  sendDisabledReason,
}: ChatComponentInputBarProps) {
  const fileInputRef = useRef<HTMLInputElement>(null)
  const canSend = Boolean(inputValue.trim()) || attachments.length > 0
  const [isComposing, setIsComposing] = useState(false)
  const { isSupported: voiceSupported, isListening, toggleListening, stopListening } = useSpeechDictation(onChangeInput)

  useEffect(() => {
    if (voiceInputDisabled && isListening) {
      stopListening()
    }
  }, [isListening, stopListening, voiceInputDisabled])

  const handleChangeInput = useCallback((value: string) => {
    if (isListening) {
      stopListening()
    }
    onChangeInput(value)
  }, [isListening, onChangeInput, stopListening])

  const handleKeyDown = useCallback<KeyboardEventHandler<HTMLTextAreaElement>>((event) => {
    const nativeEvent = event.nativeEvent as KeyboardEvent & {
      isComposing?: boolean
      keyCode?: number
    }

    if (nativeEvent.isComposing || isComposing || nativeEvent.keyCode === 229) {
      return
    }

    onKeyDown(event)
  }, [isComposing, onKeyDown])

  const handleFileInputChange = useCallback((event: ChangeEvent<HTMLInputElement>) => {
    const { files } = event.target
    if (files && files.length > 0) {
      onAttachFiles(files)
    }
    event.target.value = ''
  }, [onAttachFiles])

  const handleAttachClick = useCallback(() => {
    fileInputRef.current?.click()
  }, [])

  return (
    <div className={styles.inputContainer} data-testid="chat-composer">
      <div className={`${styles.inputWrapper} ${canSend ? styles.hasContent : ''}`}>
        {attachments.length > 0 ? (
          <div className={styles.attachmentList} data-testid="playground-attachment-list">
            {attachments.map(attachment => (
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
            ))}
          </div>
        ) : null}
        <textarea
          ref={inputRef}
          value={inputValue}
          onChange={event => handleChangeInput(event.target.value)}
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
              className={styles.attachmentFileInput}
              onChange={handleFileInputChange}
              aria-hidden="true"
              tabIndex={-1}
            />
            <button
              type="button"
              className={styles.attachButton}
              onClick={handleAttachClick}
              disabled={attachFilesDisabled || isLoading || isTogglingClawMode}
              title="Attach files (max 10 MB each)"
              aria-label="Attach files"
              data-testid="playground-attach-files"
            >
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M21.44 11.05l-9.19 9.19a6 6 0 1 1-8.49-8.49l9.19-9.19a4 4 0 1 1 5.66 5.66l-9.2 9.19a2 2 0 1 1-2.83-2.83l8.49-8.48" strokeLinecap="round" strokeLinejoin="round" />
              </svg>
            </button>
            <ToolToggle
              enabled={enableWebSearch}
              onToggle={onToggleWebSearch}
              disabled={isLoading || isTogglingClawMode}
            />
            <ClawModeToggle
              enabled={enableClawMode}
              onToggle={onToggleClawMode}
              disabled={modeToggleDisabled}
            />
            {roomChatToggleControl}
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
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                  <path
                    d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  />
                  <path d="M19 10v2a7 7 0 0 1-14 0v-2" strokeLinecap="round" strokeLinejoin="round" />
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
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                <path d="M12 19V5M5 12l7-7 7 7" strokeLinecap="round" strokeLinejoin="round" />
              </svg>
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}
