import type { ReactNode } from 'react'

import styles from './ConfigPageRouterConfigSection.module.css'

export interface Choice {
  text: string
  note?: string
  pressed: boolean
  pick: () => void
  // Why this one choice cannot be picked now, shown as its title.
  blocked?: string
}

// One segmented row of choices; the pressed one is what the listener sends.
export function Choices({ label, choices, disabled }: { label: string; choices: Choice[]; disabled: boolean }) {
  return (
    <div className={styles.vsrChoices} role="group" aria-label={label}>
      {choices.map(({ text, note, pressed, pick, blocked }) => (
        <button key={text} type="button" aria-pressed={pressed} disabled={disabled || blocked !== undefined} title={blocked} onClick={pick}>
          {text}
          {note ? <span className={styles.vsrChoiceNote}>{note}</span> : null}
        </button>
      ))}
    </div>
  )
}

export function Row({ label, hint, children }: { label: string; hint: string; children: ReactNode }) {
  return (
    <div className={styles.vsrRow}>
      <div className={styles.vsrRowText}>
        <span className={styles.vsrRowLabel}>{label}</span>
        <span className={styles.vsrRowHint}>{hint}</span>
      </div>
      {children}
    </div>
  )
}
