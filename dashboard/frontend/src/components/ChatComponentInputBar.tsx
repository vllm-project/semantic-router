import { useCallback, useState, type KeyboardEventHandler, type ReactNode, type Ref } from 'react'

import styles from './ChatComponent.module.css'
import { ClawModeToggle, ToolToggle } from './ChatComponentControls'

interface ChatComponentInputBarProps {
  enableClawMode: boolean
  enableWebSearch: boolean
  inputRef: Ref<HTMLTextAreaElement>
  inputValue: string
  isLoading: boolean
  isTogglingClawMode: boolean
  modeToggleDisabled: boolean
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
          onChange={event => onChangeInput(event.target.value)}
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
