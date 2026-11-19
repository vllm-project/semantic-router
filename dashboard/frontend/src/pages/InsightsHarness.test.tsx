import type { ReactNode } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { MemoryRouter } from 'react-router-dom'
import { describe, expect, it } from 'vitest'

import { TooltipProvider } from '../ui'

import InsightsAgentContext from './InsightsAgentContext'
import {
  apiClientCall,
  claudeRun,
  claudeSpawnCall,
  claudeSubagentCall,
  codexSubagentCall,
  deepSeekHarnessCall,
  openCodeTitleCall,
} from './insightsAgentFixtures'
import { HarnessLogo, HarnessName, RequestClassBadge, ToolChips } from './InsightsHarness'

const render = (node: ReactNode) =>
  renderToStaticMarkup(
    <MemoryRouter>
      <TooltipProvider>{node}</TooltipProvider>
    </MemoryRouter>,
  )

describe('harness logos', () => {
  it('draws each identified harness with its logo, named for assistive technology', () => {
    for (const [harness, label] of [
      ['claude_code', 'Claude Code'],
      ['codex', 'Codex'],
      ['opencode', 'OpenCode'],
      ['deepseek_harness', 'DeepSeek Harness'],
    ]) {
      const html = render(<HarnessLogo harness={harness} />)
      expect(html).toContain(`aria-label="${label}"`)
      expect(html).toContain(`data-harness="${harness}"`)
      expect(html).toContain('<img')
    }
  })

  it('inverts the currentColor logos in dark mode only', () => {
    expect(render(<HarnessLogo harness="opencode" />)).toContain('dark:invert')
    expect(render(<HarnessLogo harness="claude_code" />)).not.toContain('dark:invert')
  })

  it('falls back to a monogram for an unknown harness and an API glyph for none', () => {
    const unknown = render(<HarnessLogo harness="pi_agent" />)
    expect(unknown).toContain('aria-label="Pi Agent"')
    expect(unknown).not.toContain('<img')
    expect(unknown).toContain('>P<')
    const none = render(<HarnessName harness={undefined} />)
    expect(none).toContain('data-harness="api"')
    expect(none).toContain('API client')
  })

  it('marks a subagent with its kind and lists the tools a call used', () => {
    const badge = render(<RequestClassBadge requestClass="subagent" kind="collab_spawn" />)
    expect(badge).toContain('data-request-class="subagent"')
    expect(badge).toContain('Subagent · spawned agent')
    const chips = render(<ToolChips names={['Bash', 'Read', 'Edit', 'Grep', 'Glob']} max={3} />)
    for (const name of ['Bash', 'Read', 'Edit']) expect(chips).toContain(`>${name}<`)
    expect(chips).toContain('+2')
    expect(render(<ToolChips names={[]} />)).toBe('')
  })
})

describe('agent context of a record', () => {
  it('names a Claude Code subagent by its run label, its parent, the tools it used and how session routing held its model', () => {
    const html = render(
      <InsightsAgentContext record={claudeSubagentCall} run={claudeRun.run} turnCount={2} />,
    )
    expect(html).toContain('Claude Code')
    expect(html).toContain('subagent 1')
    expect(html).toContain('Spawned by main · #3fa9c2d8')
    expect(html).toContain('Explain the retry loop in replay_store.go')
    // the session reads as the harness sent it; the copy keeps the scoped id
    expect(html).toContain('>49b948ac-48b9-4fd5-b4e8-ccb8f6fba483<')
    expect(html).toContain('>Read<')
    expect(html).toContain('Kept the session model')
    expect(html).toContain('Session protection held the model')
    expect(html).toContain(' of 2')
  })

  it('marks the call that spawned a subagent on the main agent', () => {
    const html = render(<InsightsAgentContext record={claudeSpawnCall} run={claudeRun.run} />)
    expect(html).toContain('>main<')
    expect(html).toContain('Started a subagent on this call')
    expect(html).toContain('Picked the session model')
  })

  it('shows a Codex subagent kind and an OpenCode title call as housekeeping', () => {
    const codex = render(<InsightsAgentContext record={codexSubagentCall} />)
    expect(codex).toContain('Codex')
    expect(codex).toContain('spawned agent')
    expect(codex).toContain('thread 01a1ea71')
    const title = render(<InsightsAgentContext record={openCodeTitleCall} />)
    expect(title).toContain('OpenCode')
    expect(title).toContain('data-request-class="housekeeping"')
    expect(title).toContain('Session title')
    expect(title).toContain('Subagent reports function name from calc.py')
  })

  it('names DeepSeek Harness, and reads a request without a harness as an API client', () => {
    const dsh = render(<InsightsAgentContext record={deepSeekHarnessCall} />)
    expect(dsh).toContain('DeepSeek Harness')
    expect(dsh).toContain('No session identity on the request')
    const api = render(<InsightsAgentContext record={apiClientCall} />)
    expect(api).toContain('API client')
    expect(api).toContain('No harness or agent identity on this request')
    expect(api).toContain('data-request-class="api"')
    // no agent row without an agent identity
    expect(api).not.toContain("The run's main agent")
  })
})
