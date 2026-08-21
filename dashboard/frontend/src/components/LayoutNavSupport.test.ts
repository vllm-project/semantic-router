import { describe, expect, it } from 'vitest'

import {
  BUILD_MENU_CATEGORIES,
  findActiveLayoutMenuCategory,
  isLayoutMenuItemActive,
  PRIMARY_NAV_LINKS,
} from './LayoutNavSupport'

describe('layout navigation route matching', () => {
  it('maps named knowledge-map routes back to the Knowledge category and Bases entry', () => {
    const pathname = '/knowledge-bases/customer-support/map'
    const basesItem = BUILD_MENU_CATEGORIES.find((category) => category.key === 'knowledge')
      ?.sections.flatMap((section) => section.items)
      .find((item) => item.kind === 'route' && item.label === 'Bases')

    expect(basesItem).toBeDefined()
    expect(isLayoutMenuItemActive(basesItem!, pathname, false)).toBe(true)
    expect(findActiveLayoutMenuCategory(BUILD_MENU_CATEGORIES, pathname, false)).toBe('knowledge')
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

  it('opens Access directly on Usage and keeps the link active across Access views', () => {
    const access = PRIMARY_NAV_LINKS.find((link) => link.label === 'Access')

    expect(access).toMatchObject({ to: '/access/usage' })
    expect(access?.activePathPattern?.test('/access/api-keys')).toBe(true)
    expect(access?.activePathPattern?.test('/logs')).toBe(true)
    expect(access?.activePathPattern?.test('/config/models')).toBe(false)
  })
})
