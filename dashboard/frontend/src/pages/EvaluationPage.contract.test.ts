import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'

describe('Evaluation page enterprise contracts', () => {
  it('keeps permissions, async report states, and destructive dialogs explicit', () => {
    const source = readFileSync(new URL('./EvaluationPage.tsx', import.meta.url), 'utf8')

    expect(source).toContain('canWriteEvaluation')
    expect(source).toContain('canRunEvaluation')
    expect(source).toContain('resultsLoading')
    expect(source).toContain('resultsError')
    expect(source.match(/<ConfirmDialog/g)).toHaveLength(2)
    expect(source).not.toMatch(/\b(?:window\.)?confirm\s*\(/)
  })
})
