import { afterEach, describe, expect, it, vi } from 'vitest'

import { connectServer, deleteServer, disconnectServer, getServers, getTools } from './api'

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
