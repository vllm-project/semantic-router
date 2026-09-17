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
  it('does not present a failed Preview as a default route', () => {
    const unavailable: TestQueryResult = {
      ...result(null, false),
      isAccurate: false,
      matchedSignals: [],
      matchedDecision: null,
      warning: 'Preview unavailable: request timed out',
    }
    const html = renderToStaticMarkup(<ResultCard result={unavailable} onClose={() => {}} />)
    expect(html).toContain('Unavailable')
    expect(html).not.toContain('Default')
    expect(html).toContain('request timed out')
  })

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

it('shows a failed live preview without a fabricated default decision', () => {
  const failed = {
    ...result(null, false),
    isAccurate: false,
    matchedDecision: null,
    matchedSignals: [],
    warning: 'decision unresolved',
    signalErrors: { 'fact_check:verify': 'classifier unavailable' },
  }
  const html = renderToStaticMarkup(<ResultCard result={failed} onClose={() => {}} />)
  expect(html).toContain('decision unresolved')
  expect(html).toContain('Unavailable')
  expect(html).toContain('fact_check:verify: classifier unavailable')
  expect(html).not.toContain('Default')
})
