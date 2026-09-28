import { describe, expect, it } from 'vitest'
import { renderToStaticMarkup } from 'react-dom/server'
import { createElement } from 'react'
import { targetLabel, targetName } from './targetPresentation'
import type { Manifest, Target } from './types'
import CallEvidence from './CallEvidence'
import RecipeEvidence from './RecipeEvidence'
import TargetRequestProfile from './TargetRequestProfile'
import RunEvents from './RunEvents'
import { DEFAULT_LIMITS } from './model'

const subject: Target = {
  id: 'short-alias',
  model: 'provider/connected-model-2026',
  kind: 'single',
  base_url: 'http://localhost:8000/v1',
}
const frozen: Manifest = {
  version: 'sr-bench-1.0',
  name: 'Frozen model identity',
  mode: 'live',
  profile: 'smoke',
  seed: 1,
  limits: { ...DEFAULT_LIMITS },
  sampling: { temperature: 0, max_tokens: 16 },
  targets: [subject],
  auxiliary_targets: { judge: { ...subject, id: 'judge', model: 'provider/frozen-judge' } },
}

describe('frozen target presentation', () => {
  it('uses the canonical connected model name without inventing aliases', () => {
    expect(targetLabel(subject)).toBe('provider/connected-model-2026')
    expect(targetName(frozen, 'short-alias')).toBe(subject.model)
    expect(targetName(frozen, 'judge')).toBe('provider/frozen-judge')
  })
  it('never substitutes a current registry or target ID for missing frozen identity', () => {
    expect(targetName(frozen, 'new-registry-id')).toBe('Unknown model')
    expect(targetName(undefined, 'short-alias')).toBe('Unknown model')
    expect(targetLabel({ model: '   ' })).toBe('Unknown model')
    const changedConnection = { ...subject, model: 'changed-live-model' }
    expect(targetLabel(changedConnection)).toBe('changed-live-model')
    expect(targetName(frozen, subject.id)).toBe('provider/connected-model-2026')
  })
  it('labels fixed request profiles with the model, leaving request identity unchanged', () => {
    const original = JSON.stringify(subject)
    const html = renderToStaticMarkup(
      createElement(TargetRequestProfile, {
        target: subject,
        sampling: { temperature: 0, max_tokens: 16 },
      }),
    )
    expect(html).toContain('provider/connected-model-2026 request profile')
    expect(html).not.toContain('short-alias')
    expect(JSON.stringify(subject)).toBe(original)
  })
  it('renders case target identity from the frozen manifest and actual call model separately', () => {
    const html = renderToStaticMarkup(
      createElement(CallEvidence, {
        id: 'run',
        manifest: frozen,
        calls: [
          {
            id: 'call',
            target_id: subject.id,
            case_id: 'case',
            role: 'subject',
            status: 'completed',
            selected_model: 'actual-wire-model',
          },
        ],
        page: { total: 1, nextCursor: null, loading: false, error: '' },
        loadMore: async () => {},
      }),
    )
    expect(html).toContain('case / provider/connected-model-2026')
    expect(html).toContain('actual-wire-model')
    expect(html).not.toContain('short-alias')
  })
  it('labels the recipe using its frozen MoM model without altering snapshot keys', () => {
    const target: Target = { ...subject, kind: 'mom', model: 'connected-balance' }
    const html = renderToStaticMarkup(
      createElement(RecipeEvidence, { report: null, targets: [target] }),
    )
    expect(html).toContain('connected-balance')
    expect(html).not.toContain('short-alias')
  })
  it('resolves saved event target IDs to frozen subject and auxiliary model names', () => {
    const events = [
      { kind: 'case_running', data: { target_id: subject.id, case_id: 'case' } },
      { kind: 'call_sent', data: { target_id: 'judge', role: 'judge' } },
    ]
    const before = JSON.stringify(events)
    const html = renderToStaticMarkup(
      createElement(RunEvents, {
        manifest: frozen,
        events,
        page: { total: 2, nextCursor: null, loading: false, error: '' },
        loadMore: async () => {},
      }),
    )
    expect(html).toContain('Model: provider/connected-model-2026')
    expect(html).toContain('Model: provider/frozen-judge')
    expect(html).not.toContain('short-alias')
    expect(JSON.stringify(events)).toBe(before)
  })
})
