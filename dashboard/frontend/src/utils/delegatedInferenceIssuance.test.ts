import { describe, expect, it } from 'vitest'

import { DelegatedInferenceIssuanceIntents } from './delegatedInferenceIssuance'

describe('delegated inference issuance idempotency', () => {
  it('reuses one operation identity after response loss and rotates only after success', () => {
    let sequence = 0
    const intents = new DelegatedInferenceIssuanceIntents(() => `intent-${++sequence}`)

    const committedButLost = intents.keyFor('key-a')
    expect(intents.keyFor('key-a')).toBe(committedButLost)

    intents.complete('key-a', committedButLost)
    expect(intents.keyFor('key-a')).toBe('intent-2')
  })

  it('isolates issuance identity when the selected inference key changes', () => {
    let sequence = 0
    const intents = new DelegatedInferenceIssuanceIntents(() => `intent-${++sequence}`)

    expect(intents.keyFor('key-a')).toBe('intent-1')
    expect(intents.keyFor('key-b')).toBe('intent-2')
    intents.complete('key-a', 'intent-1')
    expect(intents.keyFor('key-b')).toBe('intent-2')

    intents.reset()
    expect(intents.keyFor('key-b')).toBe('intent-3')
  })
})
