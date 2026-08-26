import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'

const primaryLoadingSurfaces = [
  '../pages/AuthTransitionPage.tsx',
  '../pages/InviteAcceptPage.tsx',
  '../pages/DashboardPage.tsx',
  '../pages/StatusPage.tsx',
  '../pages/DslEditorPage.tsx',
  '../pages/ConfigPageModelsSection.tsx',
  '../pages/ConfigPageMoMRoutingPanel.tsx',
  '../pages/ConfigPageRoutingScopeState.tsx',
  '../pages/builderPageVisualShell.tsx',
  '../pages/topology/TopologyPageEnhanced.tsx',
  '../pages/InsightsPage.tsx',
  '../pages/InsightsRecordPage.tsx',
  '../pages/AccessControlWorkspace.tsx',
  '../pages/RequestLogDetail.tsx',
  '../pages/APIKeyDetail.tsx',
  '../pages/EvaluationPage.tsx',
  '../pages/OpenClawDashboardTab.tsx',
  './AgentManagementPanel.tsx',
  './ClawRoomChat.tsx',
  './EmbeddedServicePage.tsx',
] as const

describe('page-level loading treatment', () => {
  it.each(primaryLoadingSurfaces)('%s uses the shared product loading state', (path) => {
    const source = readFileSync(new URL(path, import.meta.url), 'utf8')

    expect(source).toContain('ProductLoadingState')
    expect(source).toMatch(/<ProductLoadingState\b/)
  })

  it('does not restore the retired page-level loading treatments', () => {
    const topology = readFileSync(
      new URL('../pages/topology/TopologyPageEnhanced.tsx', import.meta.url),
      'utf8',
    )
    const insights = readFileSync(new URL('../pages/InsightsPage.tsx', import.meta.url), 'utf8')
    const access = readFileSync(
      new URL('../pages/AccessControlWorkspace.tsx', import.meta.url),
      'utf8',
    )
    const openClaw = readFileSync(
      new URL('../pages/OpenClawDashboardTab.tsx', import.meta.url),
      'utf8',
    )

    expect(topology).not.toContain('styles.spinner')
    expect(insights).not.toContain('overviewLoading')
    expect(access).not.toContain('skeletonGrid')
    expect(openClaw).not.toContain('styles.spinner')
  })
})
