import { useEffect, useState } from 'react'
import styles from './ThinkingAnimation.module.css'

interface ThinkingAnimationProps {
  onComplete?: () => void
}

// Characters to randomly display (numbers, symbols, letters)
const CHARS = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!@#$%^&*()_+-=[]{}|;:,.<>?/~`'
const GRID_SIZE = 120 // Number of characters to display

const ThinkingAnimation = ({ onComplete }: ThinkingAnimationProps) => {
  const [characters, setCharacters] = useState<string[]>([])

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
          vLLM Semantic Router is Analyzing Your Request...
        </div>
      </div>
    </div>
  )
}

export default ThinkingAnimation

