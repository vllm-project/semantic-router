import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'

const source = readFileSync(new URL('./InferenceKeySelector.tsx', import.meta.url), 'utf8')

describe('InferenceKeySelector product contract', () => {
  it('uses a bounded server-search cursor combobox instead of materializing every key', () => {
    expect(source).toContain('role="combobox"')
    expect(source).toContain('fetchSelfInferenceKeyPage({ search }')
    expect(source).toContain('{ search, cursor: nextCursor }')
    expect(source).toContain('SELF_KEY_RENDER_LIMIT')
    expect(source).toContain('Load more')
    expect(source).not.toContain('<select')
  })

  it('debounces and aborts search while keeping keyboard and focus recovery explicit', () => {
    expect(source).toContain('SELF_KEY_SEARCH_DEBOUNCE_MS')
    expect(source).toContain('new AbortController()')
    expect(source).toContain("event.key === 'Escape'")
    expect(source).toContain("event.key === 'ArrowDown'")
    expect(source).toContain('inputRef.current?.focus()')
    expect(source).toContain('aria-activedescendant')
  })
})
