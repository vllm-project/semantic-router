import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'

describe('decision model quality editor', () => {
  it('preserves and edits decision-scoped quality scores', () => {
    const source = readFileSync(
      new URL('./ConfigPageDecisionsSection.tsx', import.meta.url),
      'utf8',
    )

    expect(source).toContain('Decision quality score')
    expect(source).toContain("| 'quality_score'")
    expect(source).toContain('modelRef.quality_score = modelRefValue.quality_score')
  })
})
