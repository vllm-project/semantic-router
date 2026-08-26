import { describe, expect, it } from 'vitest'

import {
  ANALYZE_MENU_CATEGORIES,
  BUILD_MENU_CATEGORIES,
  configSectionPath,
  filterLayoutMenuCategories,
  PRIMARY_NAV_LINKS,
  SYSTEM_MENU_CATEGORIES,
  shouldHighlightPrimaryNav,
  shouldHighlightWorkflowNav,
  WORKFLOW_NAV_LABELS,
} from './LayoutNavSupport'
import { canAccessDashboardPath } from '../utils/accessControl'

describe('layout navigation route matching', () => {
  it('uses one product label for the System workflow on every navigation surface', () => {
    expect(WORKFLOW_NAV_LABELS).toEqual({ build: 'Build', system: 'System' })
    expect(Object.values(WORKFLOW_NAV_LABELS)).not.toContain('Operate')
  })

  it('gives an open workflow menu exclusive navigation emphasis', () => {
    expect(shouldHighlightPrimaryNav(true, 'build')).toBe(false)
    expect(shouldHighlightWorkflowNav('build', false, 'build')).toBe(true)
    expect(shouldHighlightWorkflowNav('system', true, 'build')).toBe(false)
    expect(shouldHighlightPrimaryNav(true, null)).toBe(true)
    expect(shouldHighlightWorkflowNav('build', true, null)).toBe(true)
  })

  it('carries the selected Recipe only between Recipe-scoped routing editors', () => {
    expect(configSectionPath('decisions', '?recipe=recipe%2Fbalanced&view=compact')).toBe(
      '/config/decisions?recipe=recipe%2Fbalanced',
    )
    expect(configSectionPath('projections', '?recipe=recipe%2Fbalanced')).toBe(
      '/config/projections?recipe=recipe%2Fbalanced',
    )
    expect(configSectionPath('signals', '')).toBe('/config/signals')
    expect(configSectionPath('models', '?recipe=recipe%2Fbalanced')).toBe('/config/models')
  })

  it('keeps retired knowledge-base controls out of navigation', () => {
    expect(BUILD_MENU_CATEGORIES.some((category) => category.key === 'knowledge')).toBe(false)
  })

  it('groups model configuration, routing logic, and design tools into three columns', () => {
    const sections = BUILD_MENU_CATEGORIES.find((category) => category.key === 'routing')?.sections

    expect(
      sections?.map((section) => ({
        title: section.title,
        items: section.items.map((item) => item.label),
      })),
    ).toEqual([
      { title: 'Models', items: ['Models', 'Mixture-of-Models'] },
      { title: 'Routing Logic', items: ['Signals', 'Projections', 'Decisions'] },
      { title: 'Design', items: ['Brain Topology', 'DSL Builder', 'Insights'] },
    ])

    expect(sections?.[2].items[1]).toMatchObject({
      kind: 'route',
      label: 'DSL Builder',
      to: '/builder',
    })
  })

  it('links the OpenClaw integration to its canonical product route', () => {
    const integrationItems = BUILD_MENU_CATEGORIES.find(
      (category) => category.key === 'integrations',
    )?.sections.flatMap((section) => section.items)

    expect(integrationItems?.find((item) => item.label === 'OpenClaw')).toMatchObject({
      kind: 'route',
      to: '/openclaw',
    })
  })

  it('opens Access directly on Usage and keeps the link active across Access views', () => {
    const access = PRIMARY_NAV_LINKS.find((link) => link.label === 'Access')

    expect(access).toMatchObject({ to: '/access/usage' })
    expect(access?.activePathPattern?.test('/access/api-keys')).toBe(true)
    expect(access?.activePathPattern?.test('/logs')).toBe(true)
    expect(access?.activePathPattern?.test('/config/models')).toBe(false)
  })

  it('gives every visible navigation action a restrained product icon', () => {
    expect(PRIMARY_NAV_LINKS.every((link) => Boolean(link.icon))).toBe(true)
    expect(
      BUILD_MENU_CATEGORIES.flatMap((category) => category.sections)
        .flatMap((section) => section.items)
        .every((item) => Boolean(item.icon)),
    ).toBe(true)
  })

  it('does not expose the retired Dashboard Global Config editor', () => {
    const labels = BUILD_MENU_CATEGORIES.flatMap((category) => category.sections).flatMap(
      (section) => section.items.map((item) => item.label),
    )

    expect(labels).not.toContain('Global Config')
  })

  it('keeps the lowest consumer out of System while preserving read-only Routing', () => {
    const consumer = {
      role: 'read',
      permissions: ['config.read', 'topology.read', 'tools.use'],
      managementPermissions: [
        'agent.read',
        'agent.use',
        'access_policy.read',
        'delegation.use',
        'key.read',
        'quota.read',
        'routing_context.read',
        'team.read',
        'usage.read',
        'user.read',
        'tool.invoke',
        'tool.read',
      ],
    }
    const visibleBuild = filterLayoutMenuCategories(BUILD_MENU_CATEGORIES, (item) =>
      canAccessDashboardPath(
        consumer,
        item.kind === 'config' ? `/config/${item.configSection}` : item.to,
      ),
    )
    const visibleSystem = filterLayoutMenuCategories(
      [...ANALYZE_MENU_CATEGORIES, ...SYSTEM_MENU_CATEGORIES],
      (item) =>
        canAccessDashboardPath(
          consumer,
          item.kind === 'config' ? `/config/${item.configSection}` : item.to,
        ),
    )

    expect(visibleBuild.map((category) => category.key)).toEqual(['routing'])
    expect(
      visibleBuild[0].sections.flatMap((section) => section.items.map((item) => item.label)),
    ).toEqual(['Mixture-of-Models', 'Brain Topology'])
    expect(visibleSystem).toEqual([])
  })

  it('removes OpenClaw as soon as a downgraded permission snapshot is applied', () => {
    const visibleLabels = (permissions: string[]) =>
      filterLayoutMenuCategories(BUILD_MENU_CATEGORIES, (item) =>
        canAccessDashboardPath(
          { permissions, managementPermissions: ['routing.read'] },
          item.kind === 'config' ? `/config/${item.configSection}` : item.to,
        ),
      ).flatMap((category) =>
        category.sections.flatMap((section) => section.items.map((item) => item.label)),
      )

    expect(visibleLabels(['openclaw.read'])).toContain('OpenClaw')
    expect(visibleLabels([])).not.toContain('OpenClaw')
  })
})
