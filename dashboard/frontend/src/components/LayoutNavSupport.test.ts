import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'

import {
  BUILD_MENU_CATEGORIES,
  findActiveLayoutMenuCategory,
  getConfigSectionFromPathname,
  isLayoutMenuItemActive,
  OPERATE_MENU_CATEGORIES,
} from './LayoutNavSupport'

describe('layout navigation route matching', () => {
  it('closes an open workflow menu before a primary route is revealed', () => {
    const layout = readFileSync(new URL('./Layout.tsx', import.meta.url), 'utf8')
    const topNavRenderer = layout.slice(
      layout.indexOf('const renderTopNavLink'),
      layout.indexOf('const renderDesktopDropdown'),
    )

    expect(topNavRenderer).toContain('onClick={closeMenus}')
    expect(layout).toMatch(
      /useEffect\(\(\) => \{[\s\S]*setOpenDropdown\(null\)[\s\S]*\}, \[location\.pathname\]\)/,
    )
  })

  it('maps named knowledge-map routes back to the Knowledge category and Bases entry', () => {
    const pathname = '/knowledge-bases/customer-support/map'
    const basesItem = BUILD_MENU_CATEGORIES.find((category) => category.key === 'knowledge')
      ?.sections.flatMap((section) => section.items)
      .find((item) => item.kind === 'route' && item.label === 'Bases')

    expect(basesItem).toBeDefined()
    expect(isLayoutMenuItemActive(basesItem!, pathname, false)).toBe(true)
    expect(findActiveLayoutMenuCategory(BUILD_MENU_CATEGORIES, pathname, false)).toBe('knowledge')
  })

  it('keeps Model Hub, Models, and Mixture-of-Models together in the first Routing column', () => {
    const models = BUILD_MENU_CATEGORIES.find(
      (category) => category.key === 'routing',
    )?.sections.find((section) => section.title === 'Models')
    const entrypoints = models?.items.find(
      (item) => item.kind === 'config' && item.configSection === 'entrypoints-recipes',
    )

    expect(models?.items[0]).toMatchObject({
      kind: 'route',
      label: 'Model Hub',
      to: '/models',
    })
    expect(models?.items[1]).toMatchObject({
      kind: 'config',
      label: 'Models',
      configSection: 'models',
    })
    expect(entrypoints).toMatchObject({
      kind: 'config',
      label: 'Mixture-of-Models',
      configSection: 'entrypoints-recipes',
    })
    expect(models?.items.indexOf(entrypoints!)).toBe(2)
  })

  it('derives config selection from the URL, including legacy aliases', () => {
    expect(getConfigSectionFromPathname('/config/signals')).toBe('signals')
    expect(getConfigSectionFromPathname('/config/routes')).toBe('decisions')
    expect(getConfigSectionFromPathname('/config')).toBe('global-config')
    expect(getConfigSectionFromPathname('/config/not-a-section')).toBeUndefined()
    expect(getConfigSectionFromPathname('/dashboard')).toBeUndefined()
  })

  it('exposes the deployed configuration schema reference under platform operations', () => {
    const schemaReference = OPERATE_MENU_CATEGORIES.find(
      (category) => category.key === 'platform-access',
    )
      ?.sections.flatMap((section) => section.items)
      .find((item) => item.kind === 'route' && item.to === '/config/reference')

    expect(schemaReference).toMatchObject({ label: 'Schema Reference', icon: 'code' })
    expect(isLayoutMenuItemActive(schemaReference!, '/config/reference', true)).toBe(true)
  })

  it('links directly to the running Router OpenAPI UI without making Dashboard the contract owner', () => {
    const routerAPI = OPERATE_MENU_CATEGORIES.find(
      (category) => category.key === 'platform-access',
    )
      ?.sections.flatMap((section) => section.items)
      .find((item) => item.kind === 'route' && item.to === '/api/router/docs')

    expect(routerAPI).toMatchObject({
      label: 'Router API Docs',
      reloadDocument: true,
      target: '_blank',
    })
  })
})
