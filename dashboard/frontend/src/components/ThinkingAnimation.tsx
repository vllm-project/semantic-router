import { ThinkingOrb } from 'thinking-orbs'

import styles from './ThinkingAnimation.module.css'

interface ThinkingAnimationProps {
  thinkingProcess?: string
}

export default function ThinkingAnimation({ thinkingProcess }: ThinkingAnimationProps) {
  const label = thinkingProcess ? 'Working through it' : 'Thinking'

  return (
    <div className={styles.row} role="status" aria-live="polite" data-testid="chat-thinking">
      <ThinkingOrb state="composing" size={20} theme="dark" aria-label={label} />
      <span>{label}</span>
    </div>
  )
}
