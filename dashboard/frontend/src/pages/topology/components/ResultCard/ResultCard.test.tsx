import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'
import type { TestQueryResult } from '../../types'
import { ResultCard } from './ResultCard'

const result = (score: number | null, available: boolean): TestQueryResult => ({
  query: 'fixture',
  mode: 'dry-run',
  isAccurate: true,
  matchedSignals: [{ type: 'jailbreak', name: 'guard', matched: true, score, confidenceAvailable: available }],
  matchedDecision: 'block',
  matchedModels: [],
  highlightedPath: [],
  decisionConfidence: null,
  decisionConfidenceAvailable: false,
})

describe('unavailable routing confidence', () => {
  it('renders unknown score without a fabricated percentage', () => {
    const html = renderToStaticMarkup(<ResultCard result={result(null, false)} onClose={() => {}} />)
    expect(html).toContain('Score unavailable')
    expect(html).not.toContain('Score 0%')
    expect(html).not.toContain('Score 100%')
  })

  it('preserves a reported zero score', () => {
    const html = renderToStaticMarkup(<ResultCard result={result(0, true)} onClose={() => {}} />)
    expect(html).toContain('Score 0%')
  })
})
