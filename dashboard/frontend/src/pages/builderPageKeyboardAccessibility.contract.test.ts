import { readFileSync } from 'node:fs'

import { describe, expect, it } from 'vitest'

/**
 * Collapsible section headers were plain `<div onClick>` elements, so keyboard
 * users could not expand or collapse those regions (#3515). These assertions
 * pin the headers to a real button with an expanded state that names the
 * region it controls, so a revert to a static element fails here rather than
 * in review.
 */

const read = (relativePath: string): string =>
  readFileSync(new URL(relativePath, import.meta.url), 'utf8')

const collapsibleHeaders: ReadonlyArray<{
  name: string
  source: string
  expandedState: string
}> = [
  {
    name: 'web search card',
    source: '../components/ChatComponentWebToolCards.tsx',
    expandedState: 'isExpanded',
  },
  {
    name: 'config editor outline group',
    source: './editor/EditorOutline.tsx',
    expandedState: 'open',
  },
  {
    name: 'DSL guide section',
    source: '../components/DslGuide.tsx',
    expandedState: 'expanded',
  },
]

function openingTagWith(contents: string, attribute: string): string {
  const index = contents.indexOf(attribute)
  const start = contents.lastIndexOf('<', index)
  const end = contents.indexOf('>', index)
  return contents.slice(start, end)
}

describe('collapsible headers are keyboard operable', () => {
  it.each(collapsibleHeaders)(
    'renders the $name header as a button carrying its expanded state',
    ({ source, expandedState }) => {
      const contents = read(source)
      const attribute = `aria-expanded={${expandedState}}`

      expect(contents).toContain(attribute)
      const openingTag = openingTagWith(contents, attribute)
      expect(openingTag.startsWith('<button')).toBe(true)
      // The button itself names the region it controls.
      expect(openingTag).toContain('aria-controls=')
      expect(contents).not.toMatch(/<div[^>]*onClick=\{\(\) => set(Open|Expanded)/)
    },
  )

  it('wires every generated id to a region and to a control', () => {
    for (const { source } of collapsibleHeaders) {
      const contents = read(source)
      const generated = [...contents.matchAll(/const (\w+) = (?:React\.)?useId\(\)/g)].map(
        (match) => match[1],
      )

      expect(generated.length).toBeGreaterThan(0)
      for (const id of generated) {
        expect(contents).toContain(`id={${id}}`)
        const references = contents.split(`{${id}}`).length - 1
        expect(references).toBeGreaterThan(1)
      }
    }
  })
})

describe('config editor outline entries are keyboard operable', () => {
  const contents = read('./editor/EditorOutline.tsx')

  it('renders the overview and every entity entry as a button with its current state', () => {
    expect(contents).not.toMatch(/<(div|li|span)\b[^>]*\bonClick=/)
    const entries = contents.match(
      /<button\s+type="button"[^>]*?aria-current=\{active \? 'page' : undefined\}/gs,
    )
    expect(entries?.length ?? 0).toBeGreaterThanOrEqual(1)
    expect(contents).toContain("aria-current={active ? 'page' : undefined}")
  })

  it('keeps the add action outside the group toggle, so buttons never nest', () => {
    const toggleStart = contents.indexOf('aria-expanded={open}')
    const toggleEnd = contents.indexOf('</button>', toggleStart)

    expect(toggleStart).toBeGreaterThan(-1)
    expect(contents.slice(toggleStart, toggleEnd)).not.toContain('<button')
    expect(contents.slice(toggleStart, toggleEnd)).not.toContain('<AddButton')
  })
})
