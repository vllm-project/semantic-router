import { describe, expect, it } from 'vitest'

import { responseErrorMessage } from './configPageRequestErrors'

const failing = (status: number, body: string) =>
  new Response(body, { status, statusText: 'Server Error' })

describe('responseErrorMessage', () => {
  it('returns the message field from a JSON error body', async () => {
    const message = await responseErrorMessage(
      failing(500, JSON.stringify({ message: 'Failed to read config: bad field' })),
    )
    expect(message).toBe('Failed to read config: bad field')
  })

  it('returns the error field from a JSON error body', async () => {
    const message = await responseErrorMessage(
      failing(503, JSON.stringify({ error: 'router unavailable' })),
    )
    expect(message).toBe('router unavailable')
  })

  it('prefers message over error when both are present', async () => {
    const message = await responseErrorMessage(
      failing(500, JSON.stringify({ message: 'primary', error: 'secondary' })),
    )
    expect(message).toBe('primary')
  })

  it('falls back to the raw text for a JSON body without either field', async () => {
    const message = await responseErrorMessage(
      failing(500, JSON.stringify({ detail: 'unusable shape' })),
    )
    expect(message).toBe('{"detail":"unusable shape"}')
  })

  it('uses a non-JSON body as-is', async () => {
    const message = await responseErrorMessage(failing(502, 'upstream is down'))
    expect(message).toBe('upstream is down')
  })

  it('falls back to the status line when the body is empty', async () => {
    const message = await responseErrorMessage(failing(404, ''))
    expect(message).toBe('HTTP 404: Server Error')
  })
})
