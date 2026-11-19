import { describe, expect, it } from 'vitest'

import {
  API_CLIENT_LABEL,
  harnessInfo,
  normalizeHarnessId,
  PRIMARY_HARNESSES,
  recordHarnessId,
  sortHarnessIds,
} from './insightsHarness'
import {
  apiClientCall,
  claudeSpawnCall,
  codexCall,
  deepSeekHarnessCall,
  openCodeCall,
} from './insightsAgentFixtures'

describe('harness registry', () => {
  it('names the harnesses the Router identifies, each with its lobehub logo', () => {
    expect(PRIMARY_HARNESSES.map((id) => harnessInfo(id)?.label)).toEqual([
      'Claude Code',
      'Codex',
      'OpenCode',
      'Cursor',
      'OpenClaw',
      'DeepSeek Harness',
    ])
    // Vite inlines a small SVG as a data URI; the icon's own title names it.
    const title = (id: string) =>
      decodeURIComponent(harnessInfo(id)?.logo ?? '').match(/<title>(.*?)<\/title>/)?.[1]
    expect(title('claude_code')).toBe('Claude Code')
    expect(title('codex')).toBe('Codex')
    expect(title('opencode')).toBe('opencode')
    expect(title('deepseek_harness')).toBe('DeepSeek')
    expect(title('openclaw')).toBe('OpenClaw')
  })

  it('marks the logos drawn in currentColor so dark mode inverts them', () => {
    expect(harnessInfo('opencode')?.monochrome).toBe(true)
    expect(harnessInfo('cursor')?.monochrome).toBe(true)
    expect(harnessInfo('claude_code')?.monochrome).toBe(false)
    expect(harnessInfo('deepseek_harness')?.monochrome).toBe(false)
  })

  it('reads aliases and spelling variants as the canonical id', () => {
    expect(normalizeHarnessId(' Claude-Code ')).toBe('claude_code')
    expect(normalizeHarnessId('deepseek-harness')).toBe('deepseek_harness')
    expect(normalizeHarnessId('dsh')).toBe('deepseek_harness')
    expect(normalizeHarnessId('Gemini CLI')).toBe('gemini_cli')
    expect(normalizeHarnessId('roocode')).toBe('roo_code')
  })

  it('keeps an unknown harness by its own name without a logo, and none as an API client', () => {
    expect(harnessInfo('pi_agent')).toEqual({
      id: 'pi_agent',
      label: 'Pi Agent',
      monochrome: false,
      known: false,
    })
    expect(harnessInfo('')).toBeUndefined()
    expect(harnessInfo(undefined)).toBeUndefined()
    expect(API_CLIENT_LABEL).toBe('API client')
  })

  it("reads a record's harness from its agent identity only", () => {
    expect(recordHarnessId(claudeSpawnCall)).toBe('claude_code')
    expect(recordHarnessId(codexCall)).toBe('codex')
    expect(recordHarnessId(openCodeCall)).toBe('opencode')
    expect(recordHarnessId(deepSeekHarnessCall)).toBe('deepseek_harness')
    // a cc-start session is the Router's own conversation key, not a harness
    expect(recordHarnessId(apiClientCall)).toBe('')
  })

  it('lists the identified harnesses first, then the rest by label', () => {
    expect(sortHarnessIds(['goose', 'opencode', 'amp', 'claude_code', 'opencode'])).toEqual([
      'claude_code',
      'opencode',
      'amp',
      'goose',
    ])
  })
})
