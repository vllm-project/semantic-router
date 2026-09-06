import { describe, expect, it } from 'vitest'

import {
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

  it('keeps Models and Mixture-of-Models together in the first Routing column', () => {
    const models = BUILD_MENU_CATEGORIES.find(
      (category) => category.key === 'routing',
    )?.sections.find((section) => section.title === 'Models')
    const entrypoints = models?.items.find(
      (item) => item.kind === 'config' && item.configSection === 'entrypoints-recipes',
    )

    expect(models?.items[0]).toMatchObject({
      kind: 'config',
      label: 'Models',
      configSection: 'models',
    })
    expect(entrypoints).toMatchObject({
      kind: 'config',
      label: 'Mixture-of-Models',
      configSection: 'entrypoints-recipes',
    })
    expect(models?.items.indexOf(entrypoints!)).toBe(1)
  })
})
