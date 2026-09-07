import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'

import { compactModelHubLayoutQuery } from './modelHubPageController'

describe('model hub responsive detail contract', () => {
  it('opens the modal at the same breakpoint where the two-column workspace collapses', () => {
    const stylesheet = readFileSync(new URL('./ModelHubPage.module.css', import.meta.url), 'utf8')

    expect(compactModelHubLayoutQuery).toBe('(max-width: 1050px)')
    expect(stylesheet).toMatch(
      /@media \(max-width: 1050px\)[\s\S]*?\.workspace \{[\s\S]*?grid-template-columns: 1fr;[\s\S]*?\.detailLayer \{[\s\S]*?position: fixed;/,
    )
  })
})
