import { readFileSync } from 'node:fs'

import { describe, expect, it } from 'vitest'

/**
 * Collapsible section headers were plain `<div onClick>` elements, so keyboard
 * users could not expand or collapse those regions (#3515). These assertions
 * pin the headers to a real button with an expanded state, so a revert to a
 * static element fails here rather than in review.
 */

const read = (relativePath: string): string =>
  readFileSync(new URL(relativePath, import.meta.url), 'utf8')

const collapsibleHeaders: ReadonlyArray<{
  name: string
  source: string
  className: string
  expandedState: string
}> = [
  {
    name: 'web search card',
    source: '../components/ChatComponentWebToolCards.tsx',
    className: 'styles.webSearchHeader',
    expandedState: 'isExpanded',
  },
  {
    name: 'expression builder toolbox',
    source: '../components/ExpressionBuilderToolbox.tsx',
    className: 'styles.toolboxHeader',
    expandedState: '!collapsed',
  },
  {
    name: 'builder sidebar section',
    source: './builderPageDashboardViews.tsx',
    className: 'styles.sidebarSectionToggle',
    expandedState: 'open',
  },
  {
    name: 'builder validation panel',
    source: './builderPageValidationPanel.tsx',
    className: 'styles.validationHeader',
    expandedState: 'validationOpen',
  },
  {
    name: 'global settings safety section',
    source: './builderPageGlobalSettingsSafetySection.tsx',
    className: 'styles.gsSectionHeader',
    expandedState: 'expanded',
  },
  {
    name: 'topology decision node rules',
    source: './topology/components/CustomNodes/DecisionNode.tsx',
    className: 'styles.rulesHeader',
    expandedState: 'hasRuleDetail ? !rulesCollapsed : undefined',
  },
]

describe('collapsible headers are keyboard operable', () => {
  it.each(collapsibleHeaders)(
    'renders the $name header as a button carrying its expanded state',
    ({ source, className, expandedState }) => {
      const contents = read(source)

      expect(contents).toContain(`className={${className}}`)
      expect(contents).toContain(`aria-expanded={${expandedState}}`)
      expect(contents).not.toMatch(
        new RegExp(`<div[^>]*className=\\{${className.replace('.', '\\.')}\\}[^>]*onClick`)
      )

      // The button element itself must name the region it controls, so removing
      // aria-controls fails here even when the id is still referenced elsewhere.
      const buttonStart = contents.lastIndexOf('<button', contents.indexOf(`className={${className}}`))
      const buttonEnd = contents.indexOf('>', contents.indexOf(`className={${className}}`))
      const openingTag = contents.slice(buttonStart, buttonEnd)

      expect(openingTag).toContain('aria-controls=')
    }
  )

  it('wires every generated id to a region and to a control', () => {
    // Headers and their regions may live in different components once a file is
    // split, so the durable invariant is the generated id itself: it must label
    // a region and be referenced by something that points at it.
    for (const { source } of collapsibleHeaders) {
      const contents = read(source)
      const generated = [...contents.matchAll(/const (\w+) = useId\(\)/g)].map(
        (match) => match[1]
      )

      expect(generated.length).toBeGreaterThan(0)
      for (const id of generated) {
        expect(contents).toContain(`id={${id}}`)
        const references = contents.split(`{${id}}`).length - 1
        expect(references).toBeGreaterThan(1)
      }
    }
  })

  it('does not reintroduce a nested button inside the sidebar section toggle', () => {
    const contents = read('./builderPageDashboardViews.tsx')
    const toggleStart = contents.indexOf('className={styles.sidebarSectionToggle}')
    const toggleEnd = contents.indexOf('</button>', toggleStart)

    expect(toggleStart).toBeGreaterThan(-1)
    expect(contents.slice(toggleStart, toggleEnd)).not.toContain('<button')
  })
})
