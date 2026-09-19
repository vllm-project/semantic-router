import { describe, expect, it } from 'vitest'
import { describeRunEvent } from './eventPresentation'

describe('saved event presentation', () => {
  it('reads nested persisted context without exposing arbitrary payloads', () => {
    const value = describeRunEvent({
      kind: 'call_sent',
      at: '2026-01-01T00:00:00Z',
      data: {
        case_id: 'case-1',
        target_id: 'balance',
        role: 'subject',
        prompt: 'private prompt',
        secret: 'do not expose',
      },
    })
    expect(value.title).toBe('Model request dispatched')
    expect(value.description).toBe('Sent to the selected target.')
    expect(value.caseID).toBe('case-1')
    expect(JSON.stringify(value)).not.toContain('private prompt')
    expect(JSON.stringify(value)).not.toContain('do not expose')
  })
  it('preserves failure context without confusing completion with correctness or cancellation request with stop', () => {
    expect(describeRunEvent({ kind: 'case_running' })).toMatchObject({
      title: 'Case running',
      description: 'The worker is processing this case.',
      group: 'cases',
    })
    expect(describeRunEvent({ kind: 'case_completed' }).description).toBe(
      'The case result was saved.',
    )
    expect(describeRunEvent({ kind: 'cancellation_requested' }).description).toBe(
      'Cancellation is pending for active requests.',
    )
    const failure = describeRunEvent({
      kind: 'failure_observed',
      data: { reason: 'Request deadline exceeded', target_id: 'single' },
    })
    expect(failure).toMatchObject({
      attention: true,
      reason: 'Request deadline exceeded',
      targetID: 'single',
    })
  })
  it('renders unknown and legacy events as text, never serialized data', () => {
    expect(
      describeRunEvent({
        type: 'adapter_checkpoint',
        case_id: 'case-2',
        opaque: { answer: 'secret' },
      }),
    ).toMatchObject({ title: 'Adapter checkpoint', caseID: 'case-2' })
    expect(describeRunEvent({ kind: 'accounting_reconciled' }).description).toContain(
      'without new model requests',
    )
    expect(describeRunEvent({ kind: 'call_replayed' }).description).toContain(
      'no new model generation',
    )
  })
})
