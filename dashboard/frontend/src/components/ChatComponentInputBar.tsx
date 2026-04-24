import { useCallback, useEffect, useState, type KeyboardEventHandler, type ReactNode, type Ref } from 'react'

import styles from './ChatComponent.module.css'
import { ClawModeToggle, ToolToggle } from './ChatComponentControls'
import { useSpeechDictation } from '../hooks/useSpeechDictation'

interface ChatComponentInputBarProps {
  enableClawMode: boolean
  enableWebSearch: boolean
  inputRef: Ref<HTMLTextAreaElement>
  inputValue: string
  isLoading: boolean
  isTogglingClawMode: boolean
  modeToggleDisabled: boolean
  voiceInputDisabled: boolean
  onChangeInput: (value: string) => void
  onKeyDown: KeyboardEventHandler<HTMLTextAreaElement>
  onSend: () => void
  onStop: () => void
  onToggleClawMode: () => void
  onToggleWebSearch: () => void
  roomChatToggleControl: ReactNode
}

export default function ChatComponentInputBar({
  enableClawMode,
  enableWebSearch,
  inputRef,
  inputValue,
  isLoading,
  isTogglingClawMode,
  modeToggleDisabled,
  voiceInputDisabled,
  onChangeInput,
  onKeyDown,
  onSend,
  onStop,
  onToggleClawMode,
  onToggleWebSearch,
  roomChatToggleControl,
}: ChatComponentInputBarProps) {
  const canSend = Boolean(inputValue.trim())
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

  return (
    <div className={styles.inputContainer} data-testid="chat-composer">
      <div className={`${styles.inputWrapper} ${inputValue.trim() ? styles.hasContent : ''}`}>
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
              disabled={!canSend}
              title="Send message"
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
