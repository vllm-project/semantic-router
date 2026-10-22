import { describe, expect, it } from 'vitest'
import { validateWasmResponse } from './wasm'

const BUILD_HINT = 'make dashboard-build-wasm'

function spaFallbackResponse(status = 200): Response {
  return new Response('<!doctype html><html><body>app</body></html>', {
    status,
    headers: { 'content-type': 'text/html' },
  })
}

describe('validateWasmResponse', () => {
  it('rejects the HTML SPA fallback served for a missing WASM artifact', () => {
    expect(() => validateWasmResponse(spaFallbackResponse())).toThrow(BUILD_HINT)
  })

  it('rejects a non-OK response with the actionable build hint', () => {
    expect(() => validateWasmResponse(spaFallbackResponse(404))).toThrow(
      /DSL compiler WASM is not built.*HTTP 404/s,
    )
  })

  it('rejects a 200 response without the application/wasm content type', () => {
    const resp = new Response(new Uint8Array([0x00, 0x61, 0x73, 0x6d]), {
      headers: { 'content-type': 'application/octet-stream' },
    })
    expect(() => validateWasmResponse(resp)).toThrow(BUILD_HINT)
  })

  it('accepts a real application/wasm response', () => {
    const resp = new Response(new Uint8Array([0x00, 0x61, 0x73, 0x6d]), {
      headers: { 'content-type': 'application/wasm' },
    })
    expect(() => validateWasmResponse(resp)).not.toThrow()
  })
})
