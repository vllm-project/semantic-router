import { afterEach, describe, expect, it, vi } from 'vitest'

import {
  connectServer,
  deleteServer,
  disconnectServer,
  executeToolStreaming,
  getServers,
  getTools,
} from './api'
import type { MCPStreamChunk } from './types'

describe('MCP API request cancellation', () => {
  afterEach(() => vi.unstubAllGlobals())

  it('forwards abort signals to catalog requests', async () => {
    const signal = new AbortController().signal
    const fetchMock = vi.fn(async (url: string) => ({
      ok: true,
      statusText: 'OK',
      json: async () => (url.endsWith('/servers') ? { servers: [] } : { tools: [] }),
    }))
    vi.stubGlobal('fetch', fetchMock)

    await Promise.all([getServers(signal), getTools(signal)])

    expect(fetchMock).toHaveBeenCalledWith('/api/mcp/servers', { signal })
    expect(fetchMock).toHaveBeenCalledWith('/api/mcp/tools', { signal })
  })

  it('forwards abort signals and encodes IDs for connection and delete mutations', async () => {
    const signal = new AbortController().signal
    const fetchMock = vi.fn(async () => ({
      ok: true,
      statusText: 'OK',
      text: async () => '',
      json: async () => ({ status: 'connected' }),
    }))
    vi.stubGlobal('fetch', fetchMock)

    await connectServer('tenant/server', signal)
    await disconnectServer('tenant/server', signal)
    await deleteServer('tenant/server', signal)

    expect(fetchMock).toHaveBeenCalledWith('/api/mcp/servers/tenant%2Fserver/connect', {
      method: 'POST',
      signal,
    })
    expect(fetchMock).toHaveBeenCalledWith('/api/mcp/servers/tenant%2Fserver/disconnect', {
      method: 'POST',
      signal,
    })
    expect(fetchMock).toHaveBeenCalledWith('/api/mcp/servers/tenant%2Fserver', {
      method: 'DELETE',
      signal,
    })
  })
})

function sseBody(parts: string[]) {
  let index = 0
  return new ReadableStream<Uint8Array>({
    pull(controller) {
      if (index >= parts.length) {
        controller.close()
        return
      }
      controller.enqueue(new TextEncoder().encode(parts[index]))
      index += 1
    },
  })
}

async function collectStream(parts: string[], signal?: AbortSignal) {
  const fetchMock = vi.fn(async () => ({
    ok: true,
    statusText: 'OK',
    body: sseBody(parts),
  }))
  vi.stubGlobal('fetch', fetchMock)

  const chunks: MCPStreamChunk[] = []
  const generator = executeToolStreaming('srv', 'echo', {}, signal)
  let step = await generator.next()
  while (!step.done) {
    chunks.push(step.value)
    step = await generator.next()
  }
  return { chunks, result: step.value }
}

describe('MCP tool streaming', () => {
  afterEach(() => vi.unstubAllGlobals())

  it('keeps a successful result when the event name and payload type agree', async () => {
    const { chunks, result } = await collectStream([
      'event: complete\n',
      'data: {"type":"complete","data":"live-repro-ok","progress":100}\n\n',
    ])

    expect(chunks).toEqual([
      expect.objectContaining({ type: 'complete', data: 'live-repro-ok', progress: 100 }),
    ])
    expect(result).toEqual({ is_streaming: true, success: true, result: 'live-repro-ok' })
  })

  it('reads data.type when the SSE event name is message', async () => {
    const { result } = await collectStream([
      'event: message\ndata: {"type":"complete","data":"kept","progress":100}\n\n',
    ])

    expect(result).toEqual({ is_streaming: true, success: true, result: 'kept' })
  })

  it('returns success false for a tool error and preserves the payload', async () => {
    const { result } = await collectStream([
      'event: error\ndata: {"type":"error","data":"tool failed"}\n\n',
    ])

    expect(result).toEqual({
      is_streaming: true,
      success: false,
      result: 'tool failed',
      error: 'tool failed',
    })
  })

  it('returns success false for a transport failure', async () => {
    const { result } = await collectStream([
      'event: error\ndata: {"type":"error","data":"Tool execution failed"}\n\n',
    ])

    expect(result).toMatchObject({ success: false, result: 'Tool execution failed' })
  })

  it('returns success false for malformed SSE', async () => {
    const { chunks, result } = await collectStream(['event: complete\ndata: {not-json\n\n'])

    expect(chunks).toEqual([expect.objectContaining({ type: 'partial', data: '{not-json' })])
    expect(result).toMatchObject({
      success: false,
      error: 'Stream ended before a complete result',
    })
  })

  it('returns success false when the stream ends before completion', async () => {
    const { result } = await collectStream([
      'event: progress\ndata: {"type":"progress","data":"working","progress":10}\n\n',
    ])

    expect(result).toMatchObject({ success: false, error: 'Stream ended before a complete result' })
  })

  it('returns success false when the stream is cancelled', async () => {
    const fetchMock = vi.fn(async () => {
      throw new DOMException('The operation was aborted', 'AbortError')
    })
    vi.stubGlobal('fetch', fetchMock)

    const generator = executeToolStreaming('srv', 'echo', {}, AbortSignal.abort())
    const step = await generator.next()

    expect(step.done).toBe(true)
    expect(step.value).toEqual({
      is_streaming: true,
      success: false,
      error: 'Streaming execution cancelled',
    })
  })
})
