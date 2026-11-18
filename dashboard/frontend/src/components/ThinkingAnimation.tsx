import { ThinkingOrb } from 'thinking-orbs'

import { LogoMark } from '../ui'

interface ThinkingAnimationProps {
  thinkingProcess?: string
}

export default function ThinkingAnimation({ thinkingProcess }: ThinkingAnimationProps) {
  const label = thinkingProcess ? 'Working' : 'Routing your message'

  return (
    <div
      className="flex min-w-0 items-center gap-3"
      role="status"
      aria-live="polite"
      data-testid="chat-thinking"
    >
      <span className="grid size-7 shrink-0 place-items-center rounded-full border border-border bg-surface shadow-xs">
        <LogoMark size={14} />
      </span>
      <span className="flex items-center gap-2 text-[13px] text-muted-foreground">
        <ThinkingOrb state="composing" size={20} aria-label={label} />
        <span>{label}</span>
      </span>
    </div>
  )
}
