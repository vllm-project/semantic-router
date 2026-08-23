import { describe, expect, it } from 'vitest'

import {
  ANALYZE_MENU_CATEGORIES,
  BUILD_MENU_CATEGORIES,
  filterLayoutMenuCategories,
  OPERATE_MENU_CATEGORIES,
  PRIMARY_NAV_LINKS,
} from './LayoutNavSupport'
import { canAccessDashboardPath } from '../utils/accessControl'

describe('layout navigation route matching', () => {
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

  it('keeps the lowest consumer out of Operate while preserving read-only Routing', () => {
    const consumer = {
      role: 'read',
      permissions: ['config.read', 'topology.read', 'tools.use'],
      managementPermissions: [
        'delegation.use',
        'key.read',
        'quota.read',
        'routing.read',
        'team.read',
        'usage.read',
        'user.read',
      ],
    }
    const visibleBuild = filterLayoutMenuCategories(BUILD_MENU_CATEGORIES, (item) =>
      canAccessDashboardPath(
        consumer,
        item.kind === 'config' ? `/config/${item.configSection}` : item.to,
      ),
    )
    const visibleOperate = filterLayoutMenuCategories(
      [...ANALYZE_MENU_CATEGORIES, ...OPERATE_MENU_CATEGORIES],
      (item) =>
        canAccessDashboardPath(
          consumer,
          item.kind === 'config' ? `/config/${item.configSection}` : item.to,
        ),
    )

    expect(visibleBuild.map((category) => category.key)).toEqual(['routing'])
    expect(
      visibleBuild[0].sections.flatMap((section) => section.items.map((item) => item.label)),
    ).toEqual([
      'Models',
      'Mixture-of-Models',
      'Signals',
      'Projections',
      'Decisions',
      'Brain Topology',
      'DSL Builder',
    ])
    expect(visibleOperate).toEqual([])
  })
})
