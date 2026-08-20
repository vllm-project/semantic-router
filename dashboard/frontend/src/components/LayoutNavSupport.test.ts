import { describe, expect, it } from 'vitest'

import {
  ACCESS_MENU_CATEGORIES,
  BUILD_MENU_CATEGORIES,
  findActiveLayoutMenuCategory,
  isLayoutMenuItemActive,
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
      { title: 'Design', items: ['Brain Topology', 'DSL Builder'] },
    ])

    expect(sections?.[2].items[1]).toMatchObject({
      kind: 'route',
      label: 'DSL Builder',
      to: '/builder',
    })
  })

  it('gives Credentials, Identity, Policy, and Observe independent Access tabs', () => {
    expect(ACCESS_MENU_CATEGORIES.map((category) => category.label)).toEqual([
      'Credentials',
      'Identity',
      'Policy',
      'Observe',
    ])
    const identity = ACCESS_MENU_CATEGORIES.find((category) => category.key === 'identity')

    expect(identity?.sections.flatMap((section) => section.items)).toContainEqual(
      expect.objectContaining({ label: 'Users', to: '/access/users' }),
    )
    expect(identity?.sections.flatMap((section) => section.items)).toContainEqual(
      expect.objectContaining({ label: 'Teams', to: '/access/teams' }),
    )
  })
})
