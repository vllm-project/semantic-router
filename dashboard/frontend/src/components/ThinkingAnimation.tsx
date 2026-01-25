import { useEffect, useState, useRef } from 'react'
import styles from './ThinkingAnimation.module.css'
import PlatformBranding from './PlatformBranding'

interface ThinkingAnimationProps {
  onComplete?: () => void
  thinkingProcess?: string
}

// Characters to randomly display (numbers, symbols, letters)
const CHARS = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!@#$%^&*()_+-=[]{}|;:,.<>?/~`'
const GRID_SIZE = 120 // Number of characters to display

const ThinkingAnimation = ({ onComplete, thinkingProcess }: ThinkingAnimationProps) => {
  const [characters, setCharacters] = useState<string[]>([])
  const thinkingContentRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    // Initialize with random characters
    setCharacters(Array.from({ length: GRID_SIZE }, () =>
      CHARS[Math.floor(Math.random() * CHARS.length)]
    ))

    // Update characters rapidly
    const interval = setInterval(() => {
      setCharacters(prev =>
        prev.map(() => CHARS[Math.floor(Math.random() * CHARS.length)])
      )
    }, 50) // Update every 50ms for fast flickering

    return () => {
      clearInterval(interval)
    }
  }, [])

  // Call onComplete when component unmounts (when parent hides it)
  useEffect(() => {
    return () => {
      onComplete?.()
    }
  }, [onComplete])

  // Auto-scroll to bottom when thinking process updates
  useEffect(() => {
    if (thinkingContentRef.current && thinkingProcess) {
      thinkingContentRef.current.scrollTop = thinkingContentRef.current.scrollHeight
    }
  }, [thinkingProcess])

  return (
    <div className={styles.overlay}>
      <div className={styles.container}>
        <div className={styles.grid}>
          {characters.map((char, index) => (
            <span
              key={index}
              className={styles.char}
              style={{
                animationDelay: `${Math.random() * 0.5}s`,
                opacity: 0.3 + Math.random() * 0.7,
              }}
            >
              {char}
            </span>
          ))}
        </div>
        <div className={styles.statusText}>
          vLLM Semantic Router is Thinking...
        </div>

        {/* Show thinking process if available */}
        {thinkingProcess && (
          <div ref={thinkingContentRef} className={styles.thinkingContent}>
            <div className={styles.thinkingLabel}>Thinking Process:</div>
            <pre className={styles.thinkingText}>{thinkingProcess}</pre>
          </div>
        )}

        {/* Option A: Platform branding below status text */}
        <PlatformBranding variant="default" />
      </div>
    </div>
  )
}

export default ThinkingAnimation

