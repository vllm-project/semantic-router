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

// The looks of outcomes that are not an app's own: a kept subagent, an unverified answer, an allowed command.
export const LOOKS = {
  kept: { glyph: '≈', color: 'warning' },
  unverified: { glyph: '!', color: 'warning' },
  allowed: { glyph: '✓', color: 'success' },
} as const satisfies Record<string, { glyph: string; color: Color }>

// In the order the surfaces list them.
export const SECTIONS = Object.keys(APPS) as readonly Section[]
export const APP_NAMES = SECTIONS.filter((section): section is App => section !== 'off')

// The app a decision belongs to: a kept subagent is the router's, a skipped call its app's.
export function appOf(d: Decision): App {
  if (d.kind === 'route' || d.kind === 'keep') return 'router'
  return d.kind === 'unavailable' ? d.app : d.kind
}

// Where a decision is listed: under its app, or with the skipped calls.
export const sectionOf = (d: Decision): Section => (d.kind === 'unavailable' ? 'off' : appOf(d))
