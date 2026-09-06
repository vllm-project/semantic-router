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
    expandedState: '!toolboxCollapsed',
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
    }
  )

  it('keeps every collapsible header attached to the region it controls', () => {
    for (const { source } of collapsibleHeaders) {
      const contents = read(source)
      // Matches both `aria-controls={id}` and `aria-controls={cond ? id : undefined}`.
      const controlled = [
        ...contents.matchAll(/aria-controls=\{(?:[A-Za-z]+ \? )?([A-Za-z]+)/g),
      ].map((match) => match[1])

      expect(controlled.length).toBeGreaterThan(0)
      for (const id of controlled) {
        expect(contents).toContain(`id={${id}}`)
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
