import { readFileSync } from 'node:fs'
import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it, vi } from 'vitest'

import { DashboardTab, StatusTab } from './OpenClawPageTabs'
import type { OpenClawStatus, TeamProfile } from './OpenClawPageSupport'
import { TeamTab } from './OpenClawTeamTab'
import { WorkerTab } from './OpenClawWorkerTab'

const team: TeamProfile = {
  id: 'routing-core',
  name: 'Routing Core',
  role: 'Operations',
}

const worker: OpenClawStatus = {
  running: false,
  containerName: 'openclaw-atlas',
  gatewayUrl: '',
  port: 18789,
  healthy: false,
  error: '',
  teamId: team.id,
  teamName: team.name,
  agentName: 'Atlas',
}

describe('enterprise permission surfaces', () => {
  it('keeps team data visible without rendering team mutation controls', () => {
    const markup = renderToStaticMarkup(
      createElement(TeamTab, {
        teams: [team],
        teamsLoading: false,
        containers: [worker],
        onTeamsUpdated: vi.fn(),
        readOnly: true,
      }),
    )

    expect(markup).toContain('Routing Core')
    expect(markup).not.toMatch(/>New Team</)
    expect(markup).not.toMatch(/>Edit</)
    expect(markup).not.toMatch(/>Delete</)
  })

  it('keeps worker status navigation while hiding worker mutation controls', () => {
    const markup = renderToStaticMarkup(
      createElement(WorkerTab, {
        containers: [worker],
        teams: [team],
        onProvisioned: vi.fn(),
        onSwitchToTeam: vi.fn(),
        onSwitchToStatus: vi.fn(),
        readOnly: true,
      }),
    )

    expect(markup).toContain('Atlas')
    expect(markup).toMatch(/>Status</)
    expect(markup).not.toMatch(/>New Worker</)
    expect(markup).not.toMatch(/>Edit</)
    expect(markup).not.toMatch(/>Delete</)
  })

  it('labels dashboard navigation as view-only instead of management', () => {
    const markup = renderToStaticMarkup(
      createElement(DashboardTab, {
        containers: [worker],
        teams: [team],
        onSwitchToStatus: vi.fn(),
        readOnly: true,
      }),
    )

    expect(markup).toContain('View Claw Status')
    expect(markup).toContain('View status')
    expect(markup).not.toMatch(/>Manage</)
  })

  it('renders runtime status and refresh without lifecycle actions', () => {
    const markup = renderToStaticMarkup(
      createElement(StatusTab, {
        containers: [worker],
        statusLoading: false,
        onRefresh: vi.fn(),
        readOnly: true,
      }),
    )

    expect(markup).toContain('openclaw-atlas')
    expect(markup).toMatch(/>Refresh Status</)
    expect(markup).not.toMatch(/>Start</)
    expect(markup).not.toMatch(/>Stop</)
    expect(markup).not.toMatch(/>Remove</)
    expect(markup).not.toMatch(/>Dashboard</)
  })

  it('keeps permission, structured-input, dialog, tab, and polling contracts explicit', () => {
    const securitySource = readFileSync(
      new URL('./SecurityPolicyPage.tsx', import.meta.url),
      'utf8',
    )
    const pageSource = readFileSync(new URL('./OpenClawPage.tsx', import.meta.url), 'utf8')
    const configSource = readFileSync(new URL('./ConfigPage.tsx', import.meta.url), 'utf8')
    const mutationSources = [
      readFileSync(new URL('./OpenClawStatusTab.tsx', import.meta.url), 'utf8'),
      readFileSync(new URL('./OpenClawTeamTab.tsx', import.meta.url), 'utf8'),
      readFileSync(new URL('./OpenClawWorkerTab.tsx', import.meta.url), 'utf8'),
    ]

    expect(securitySource).toContain('canManageSecurity')
    expect(securitySource).toContain('<StringListEditor')
    expect(securitySource).toContain('security.manage')
    expect(securitySource).not.toContain('comma-separated')
    expect(securitySource).not.toContain('modelInputs')

    expect(pageSource).toContain('canManageOpenClaw')
    expect(pageSource).toContain('openclaw.manage')
    expect(pageSource).toContain('createLatestOpenClawRequest')
    expect(pageSource).toContain('document.hidden')
    expect(pageSource).toContain("document.addEventListener('visibilitychange'")
    expect(pageSource).toContain('role="tablist"')
    expect(pageSource).toContain('role="tab"')
    expect(pageSource).toContain('role="tabpanel"')
    expect(pageSource).toContain('aria-selected=')

    expect(configSource).toContain("const isMCPSection = activeSection === 'mcp'")
    expect(configSource).toContain('{isMCPSection && (')
    expect(configSource).toContain('!canWriteConfig(user) || !canDeployConfig(user)')
    expect(configSource).toContain('config.write and config.deploy permissions')

    for (const source of mutationSources) {
      expect(source).toContain('<ConfirmDialog')
      expect(source).not.toMatch(/\b(?:window\.)?confirm\s*\(/)
    }
  })
})
