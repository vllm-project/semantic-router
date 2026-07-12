import { memo, useEffect, useRef } from 'react'
import type { CSSProperties } from 'react'
import styles from './ThinkingAnimation.module.css'
import PlatformBranding from './PlatformBranding'

interface ThinkingAnimationProps {
  onComplete?: () => void
  thinkingProcess?: string
}

const CHARS =
  '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!@#$%^&*()_+-=[]{}|;:,.<>?/~`'
const GRID_SIZE = 120
const UPDATE_INTERVAL = 72
const UPDATE_BATCH_SIZE = 20

const characterAt = (index: number, tick: number) =>
  CHARS[(index * 37 + tick * 17 + (index % 7) * tick) % CHARS.length]

const INITIAL_CHARACTERS = Array.from({ length: GRID_SIZE }, (_, index) => characterAt(index, 0))

const CharacterMatrix = memo(() => {
  const gridRef = useRef<HTMLDivElement>(null)
  const characterRefs = useRef<Array<HTMLSpanElement | null>>([])

  useEffect(() => {
    const grid = gridRef.current
    if (!grid) return

    const motionQuery = window.matchMedia('(prefers-reduced-motion: reduce)')
    let animationFrame: number | null = null
    let lastUpdate = 0
    let tick = 0

    const renderCharacters = (nextTick: number, renderAll = false) => {
      const startIndex = (nextTick * UPDATE_BATCH_SIZE) % GRID_SIZE
      const updateCount = renderAll ? GRID_SIZE : UPDATE_BATCH_SIZE

      for (let offset = 0; offset < updateCount; offset += 1) {
        const index = renderAll ? offset : (startIndex + offset) % GRID_SIZE
        const element = characterRefs.current[index]
        if (element) element.textContent = characterAt(index, nextTick)
      }
    }

    const stopAnimation = () => {
      if (animationFrame !== null) {
        cancelAnimationFrame(animationFrame)
        animationFrame = null
      }
    }

    const animate = (timestamp: number) => {
      if (timestamp - lastUpdate >= UPDATE_INTERVAL) {
        lastUpdate = timestamp
        tick += 1
        renderCharacters(tick)
      }
      animationFrame = requestAnimationFrame(animate)
    }

    const syncMotion = () => {
      stopAnimation()
      grid.dataset.motion = motionQuery.matches ? 'static' : 'animated'

      if (motionQuery.matches) {
        tick = 0
        renderCharacters(tick, true)
      } else {
        animationFrame = requestAnimationFrame(animate)
      }
    }

    motionQuery.addEventListener('change', syncMotion)
    syncMotion()

    return () => {
      stopAnimation()
      motionQuery.removeEventListener('change', syncMotion)
    }
  }, [])

  return (
    <div
      ref={gridRef}
      className={styles.grid}
      data-testid="thinking-grid"
      data-motion="animated"
      aria-hidden="true"
    >
      {INITIAL_CHARACTERS.map((character, index) => (
        <span
          key={index}
          ref={(element) => {
            characterRefs.current[index] = element
          }}
          className={`${styles.char} ${index % 29 === 0 ? styles.signalChar : ''}`}
          style={
            {
              animationDelay: `${-(index % 12) * 0.07}s`,
              animationDuration: `${0.44 + (index % 7) * 0.055}s`,
              opacity: 0.34 + (index % 5) * 0.12,
            } as CSSProperties
          }
        >
          {character}
        </span>
      ))}
    </div>
  )
})

CharacterMatrix.displayName = 'CharacterMatrix'

const ThinkingAnimation = ({ onComplete, thinkingProcess }: ThinkingAnimationProps) => {
  const thinkingContentRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    return () => {
      onComplete?.()
    }
  }, [onComplete])

  useEffect(() => {
    if (thinkingContentRef.current && thinkingProcess) {
      thinkingContentRef.current.scrollTop = thinkingContentRef.current.scrollHeight
    }
  }, [thinkingProcess])

  return (
    <div className={styles.overlay}>
      <div className={styles.container}>
        <CharacterMatrix />
        <div className={styles.statusText} role="status" aria-live="polite">
          vLLM Semantic Router is Thinking...
        </div>

        {thinkingProcess && (
          <div ref={thinkingContentRef} className={styles.thinkingContent}>
            <div className={styles.thinkingLabel}>Thinking Process:</div>
            <pre className={styles.thinkingText}>{thinkingProcess}</pre>
          </div>
        )}

        <PlatformBranding variant="default" />
      </div>
    </div>
  )
}

export default ThinkingAnimation
