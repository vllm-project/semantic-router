import { memo, useCallback, useEffect, useState } from 'react'

import styles from './ChatComponent.module.css'

const CopyResponseButton = ({ copied, onCopy }: { copied: boolean; onCopy: () => void }) => {
  return (
    <button
      className={styles.actionButton}
      onClick={onCopy}
      title={copied ? 'Copied!' : 'Copy'}
      aria-label={copied ? 'Copied!' : 'Copy'}
    >
      {copied ? (
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
          <polyline points="20 6 9 17 4 12" />
        </svg>
      ) : (
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
          <rect x="9" y="9" width="13" height="13" rx="2" />
          <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" />
        </svg>
      )}
    </button>
  )
}

export const MessageActionBar = ({ content }: { content: string }) => {
  const [copied, setCopied] = useState(false)

  const handleCopy = useCallback(async () => {
    if (!content) return
    try {
      if (navigator.clipboard && navigator.clipboard.writeText) {
        await navigator.clipboard.writeText(content)
      } else {
        const textArea = document.createElement('textarea')
        textArea.value = content
        textArea.style.position = 'fixed'
        textArea.style.left = '-9999px'
        document.body.appendChild(textArea)
        textArea.select()
        document.execCommand('copy')
        document.body.removeChild(textArea)
      }
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    } catch (err) {
      console.error('Failed to copy:', err)
    }
  }, [content])

  return (
    <div className={styles.messageActionBar}>
      <CopyResponseButton copied={copied} onCopy={handleCopy} />
    </div>
  )
}

export const TypingGreeting = memo(({ lines }: { lines: string[] }) => {
  const [currentLineIndex, setCurrentLineIndex] = useState(0)
  const [displayedText, setDisplayedText] = useState('')
  const [isTyping, setIsTyping] = useState(true)

  useEffect(() => {
    if (currentLineIndex >= lines.length) return

    const currentLine = lines[currentLineIndex]
    let charIndex = 0
    setIsTyping(true)
    setDisplayedText('')

    const typingInterval = setInterval(() => {
      if (charIndex < currentLine.length) {
        setDisplayedText(currentLine.slice(0, charIndex + 1))
        charIndex++
      } else {
        clearInterval(typingInterval)
        setIsTyping(false)
        setTimeout(() => {
          if (currentLineIndex < lines.length - 1) {
            setCurrentLineIndex(prev => prev + 1)
          }
        }, 1500)
      }
    }, 60)

    return () => clearInterval(typingInterval)
  }, [currentLineIndex, lines])

  return (
    <div className={styles.typingGreeting} translate="no">
      <h2>
        {displayedText}
        {isTyping && <span className={styles.typingCursor}>|</span>}
      </h2>
    </div>
  )
})

export const ToolToggle = ({
  enabled,
  onToggle,
  disabled
}: {
  enabled: boolean
  onToggle: () => void
  disabled?: boolean
}) => {
  return (
    <button
      className={`${styles.inputActionButton} ${enabled ? styles.searchToggleActive : ''}`}
      onClick={onToggle}
      disabled={disabled}
      data-tooltip={enabled ? 'Web Search enabled' : 'Enable Web Search'}
    >
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <circle cx="12" cy="12" r="10" />
        <path d="M2 12h20" />
        <path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z" />
      </svg>
    </button>
  )
}

export const ClawModeToggle = ({
  enabled,
  onToggle,
  disabled
}: {
  enabled: boolean
  onToggle: () => void
  disabled?: boolean
}) => {
  return (
    <button
      className={`${styles.clawModeToggleButton} ${enabled ? styles.clawToggleActive : ''}`}
      onClick={onToggle}
      disabled={disabled}
      type="button"
      aria-pressed={enabled}
      aria-label={enabled ? 'Disable HireClaw' : 'Enable HireClaw'}
    >
      <img src="/openclaw.svg" alt="" aria-hidden="true" className={styles.clawToggleIcon} />
      <span className={styles.clawToggleLabel}>HireClaw</span>
    </button>
  )
}
