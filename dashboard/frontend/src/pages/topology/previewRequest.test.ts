import { describe, expect, it } from 'vitest'
import { parsePreviewEnvelope, previewRecipeName } from './previewRequest'

it('sends the named v0.5 recipe behind the default dashboard scope', () => {
  expect(previewRecipeName('default', 'balanced')).toBe('balanced')
  expect(previewRecipeName('unbound', 'balanced')).toBe('unbound')
  expect(previewRecipeName('default')).toBe('default')
})

describe('Topology request context', () => {
  it('preserves multimodal messages, tools and metadata', () => {
    const envelope = {
      messages: [
        {
          role: 'user',
          content: [{ type: 'image_url', image_url: { url: 'https://example.com/image.png' } }],
        },
      ],
      metadata: { tenant: 'example' },
      tools: [{ type: 'function', function: { name: 'lookup', parameters: { type: 'object' } } }],
    }
    expect(parsePreviewEnvelope(JSON.stringify(envelope))).toEqual(envelope)
    expect(parsePreviewEnvelope('')).toEqual({})
  })
  it.each([
    'null',
    '[]',
    '{',
    '{"messages":{}}',
    '{"tools":[null]}',
    '{"metadata":{"tenant":3}}',
    '{"model":"other"}',
  ])('rejects invalid or silently ignored input: %s', (value) => {
    expect(() => parsePreviewEnvelope(value)).toThrow()
  })
})
