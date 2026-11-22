import type { ReactNode } from 'react'

import { cn } from '../ui'

export interface Choice {
  text: string
  note?: string
  pressed: boolean
  pick: () => void
  // Why this one choice cannot be picked now, shown as its title.
  blocked?: string
}

// One segmented row of choices; the pressed one is what the listener sends.
export function Choices({
  label,
  choices,
  disabled,
}: {
  label: string
  choices: Choice[]
  disabled: boolean
}) {
  return (
    <div
      className="inline-flex shrink-0 items-center gap-0.5 rounded-lg border border-border bg-surface-subtle p-0.5"
      role="group"
      aria-label={label}
    >
      {choices.map(({ text, note, pressed, pick, blocked }) => (
        <button
          key={text}
          className={cn(
            'inline-flex h-7 items-center gap-1 rounded-md px-2.5 text-[13px] font-medium text-muted-foreground outline-none transition-[color,background-color,box-shadow] hover:text-foreground focus-visible:ring-[3px] focus-visible:ring-ring/40 disabled:pointer-events-none disabled:opacity-50',
            'aria-pressed:bg-surface aria-pressed:text-foreground aria-pressed:shadow-sm dark:aria-pressed:bg-accent',
          )}
          type="button"
          disabled={disabled || blocked !== undefined}
          title={blocked}
          onClick={pick}
          aria-pressed={pressed}
        >
          {text}
          {note ? (
            <span className="font-mono text-[11px] font-normal text-subtle-foreground">{note}</span>
          ) : null}
        </button>
      ))}
    </div>
  )
}

export function Row({
  label,
  hint,
  children,
}: {
  label: string
  hint: string
  children: ReactNode
}) {
  return (
    <div className="flex flex-col gap-2 py-3 sm:flex-row sm:items-center sm:justify-between sm:gap-6">
      <div className="flex min-w-0 flex-col gap-0.5">
        <span className="text-[13px] font-medium">{label}</span>
        <span className="text-xs leading-relaxed text-muted-foreground">{hint}</span>
      </div>
      {children}
    </div>
  )
}
