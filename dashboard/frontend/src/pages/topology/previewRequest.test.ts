import { describe, expect, it } from 'vitest'
import { parsePreviewEnvelope } from './previewRequest'

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
