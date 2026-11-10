import { describe, expect, it } from 'vitest'

import { decisionOutputForEditor, decisionOutputFromEditor } from './configPageSupport'

describe('decision output contract in the editor', () => {
  it('edits kind as a list and writes one kind as a scalar', () => {
    expect(decisionOutputForEditor({ kind: 'image', action: 'generate' })).toEqual({
      kind: ['image'],
      action: 'generate',
    })
    expect(decisionOutputFromEditor({ kind: ['image'], action: 'generate' })).toEqual({
      kind: 'image',
      action: 'generate',
    })
    expect(decisionOutputFromEditor({ kind: ['text', ' image '] })).toEqual({
      kind: ['text', 'image'],
    })
  })

  it('drops an empty contract, so the decision answers in text', () => {
    expect(decisionOutputForEditor(undefined)).toEqual({})
    expect(decisionOutputFromEditor({})).toBeUndefined()
    expect(decisionOutputFromEditor({ kind: [], action: '' })).toBeUndefined()
  })
})
