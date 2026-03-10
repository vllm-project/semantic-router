import type { KeyboardEventHandler, ReactNode, Ref } from 'react'

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
  return (
    <div className={styles.inputContainer}>
      <div className={`${styles.inputWrapper} ${inputValue.trim() ? styles.hasContent : ''}`}>
        <textarea
          ref={inputRef}
          value={inputValue}
          onChange={event => onChangeInput(event.target.value)}
          onKeyDown={onKeyDown}
          placeholder="Ask me anything..."
          className={styles.input}
          rows={1}
          disabled={isLoading}
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
          {isLoading ? (
            <button
              className={`${styles.sendButton} ${styles.stopButton}`}
              onClick={onStop}
              title="Stop generating"
            >
              <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
                <rect x="6" y="6" width="12" height="12" rx="2" />
              </svg>
            </button>
          ) : (
            <button
              className={styles.sendButton}
              onClick={onSend}
              disabled={!inputValue.trim()}
              title="Send message"
            >
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                <path d="M12 19V5M5 12l7-7 7 7" strokeLinecap="round" strokeLinejoin="round" />
              </svg>
            </button>
          )}
        </div>
      </div>
    </div>
  )
}
