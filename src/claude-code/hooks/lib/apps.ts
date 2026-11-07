import type { ThemeKey } from 'claude-code'

import type { App, Decision } from '../../types'

export type Color = ThemeKey

export type Section = App | 'off'

// Every surface (cards, band, pane) takes an application's look from here.
export const APPS: Record<Section, { name: string; glyph: string; color: Color }> = {
  router: { name: 'router', glyph: '⇢', color: 'suggestion' },
  escalate: { name: 'escalate', glyph: '↑', color: 'claude' },
  redact: { name: 'redact', glyph: '▒', color: 'merged' },
  shield: { name: 'shield', glyph: '◆', color: 'warning' },
  guard: { name: 'guard', glyph: '‖', color: 'warning' },
  check: { name: 'check', glyph: '✓', color: 'success' },
  off: { name: 'off', glyph: '○', color: 'inactive' },
}

export const SECTIONS: readonly Section[] = ['router', 'escalate', 'redact', 'shield', 'guard', 'check', 'off']

export const sectionOf = (decision: Decision): Section => {
  switch (decision.kind) {
    case 'route':
    case 'keep':
      return 'router'
    case 'unavailable':
      return 'off'
    default:
      return decision.kind
  }
}
