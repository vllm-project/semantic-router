import { describe, expect, it } from 'vitest'

import type { MCPServerState } from '../tools/mcp'
import type { UnifiedTool } from './mcpConfigPanelTypes'
import {
  buildServerConfig,
  buildTestServerConfig,
  filterServers,
  filterUnifiedTools,
  getMCPVisibleRange,
  hiddenArgumentsBlockCommandChange,
  paginateMCPItems,
  type ServerFormValues,
} from './mcpConfigPanelUtils'

const baseFormValues: ServerFormValues = {
  name: 'Docs server',
  description: '',
  transport: 'stdio',
  enabled: true,
  command: 'npx',
  args: [],
  preserveArguments: false,
  url: '',
  headers: {},
  timeout: '30000',
}

function makeTool(index: number, sourceType: UnifiedTool['sourceType']): UnifiedTool {
  return {
    id: `${sourceType}-${index}`,
    name: `tool-${String(index).padStart(2, '0')}`,
    description: index === 17 ? 'Search private documentation' : `Tool ${index}`,
    source: sourceType === 'mcp' ? 'Docs server' : 'Semantic Router',
    sourceType,
    parameters:
      index % 3 === 0
        ? [{ name: 'tenant', type: 'string', description: 'Tenant selector', required: true }]
        : [],
    rawTool: {} as UnifiedTool['rawTool'],
  }
}

function makeServer(index: number): MCPServerState {
  const connected = index % 2 === 0
  return {
    config: {
      id: `server-${index}`,
      name: `Server ${String(index).padStart(2, '0')}`,
      description: index === 13 ? 'Private document gateway' : undefined,
      transport: 'streamable-http',
      connection: { url: `https://mcp-${index}.example.test` },
      enabled: true,
    },
    status: connected ? 'connected' : 'disconnected',
    tools: Array.from({ length: index % 5 }, (_, toolIndex) => ({
      name: `server-${index}-tool-${toolIndex}`,
      description: 'Server tool',
      inputSchema: { type: 'object' as const },
    })),
  }
}

describe('MCP configuration support', () => {
  it('builds stdio arguments and HTTP headers from structured values', () => {
    const stdio = buildServerConfig({
      ...baseFormValues,
      args: [' -y ', '', '@modelcontextprotocol/server-filesystem', ' /workspace '],
    })
    expect(stdio.connection).toEqual({
      command: 'npx',
      args: ['-y', '@modelcontextprotocol/server-filesystem', '/workspace'],
    })

    const http = buildTestServerConfig('server-1', {
      ...baseFormValues,
      transport: 'streamable-http',
      command: '',
      url: 'https://mcp.example.test',
      headers: {
        ' Authorization ': ' Bearer token:with:colons ',
        Empty: '',
      },
    })
    expect(http.connection).toEqual({
      url: 'https://mcp.example.test',
      headers: { Authorization: 'Bearer token:with:colons' },
    })

    const clearedHeaders = buildServerConfig({
      ...baseFormValues,
      transport: 'streamable-http',
      command: '',
      url: 'https://mcp.example.test',
      headers: {},
    })
    expect(clearedHeaders.connection).toEqual({
      url: 'https://mcp.example.test',
      headers: {},
    })
  })

  it('distinguishes preserving hidden arguments from explicitly clearing them', () => {
    const preserved = buildServerConfig({
      ...baseFormValues,
      preserveArguments: true,
    })
    expect(preserved.connection).toEqual({ command: 'npx', args: undefined })

    const cleared = buildServerConfig({
      ...baseFormValues,
      preserveArguments: false,
      args: [],
    })
    expect(cleared.connection).toEqual({ command: 'npx', args: [] })
  })

  it('requires hidden arguments to be resolved before changing a stdio command', () => {
    const server: MCPServerState['config'] = {
      id: 'filesystem',
      name: 'Filesystem',
      transport: 'stdio',
      enabled: true,
      connection: { command: 'npx' },
    }
    expect(hiddenArgumentsBlockCommandChange(server, 'new-command', true)).toBe(true)
    expect(hiddenArgumentsBlockCommandChange(server, 'new-command', false)).toBe(false)
    expect(hiddenArgumentsBlockCommandChange(server, 'npx', true)).toBe(false)
  })

  it('searches, filters, sorts by name, and bounds a large tools catalog', () => {
    const tools = Array.from({ length: 48 }, (_, index) =>
      makeTool(index, index % 3 === 0 ? 'mcp' : index % 3 === 1 ? 'frontend' : 'backend'),
    ).reverse()
    const searched = filterUnifiedTools(tools, 'private documentation', 'all')
    expect(searched.map((tool) => tool.id)).toEqual(['backend-17'])

    const mcpTools = filterUnifiedTools(tools, 'tenant', 'mcp')
    expect(mcpTools).toHaveLength(16)
    expect(mcpTools.every((tool) => tool.sourceType === 'mcp')).toBe(true)
    expect(mcpTools[0]?.name).toBe('tool-00')
    tools.reverse()
    expect(paginateMCPItems(tools, 2, 12)).toHaveLength(12)
    expect(paginateMCPItems(tools, 2, 12)[0]?.id).toBe('mcp-12')
    expect(getMCPVisibleRange(48, 2, 12)).toEqual({ start: 13, end: 24 })
  })

  it('searches, filters by transport and status, and bounds servers in name order', () => {
    const servers = Array.from({ length: 25 }, (_, index) => makeServer(index)).reverse()
    const none = { transports: [], statuses: [] }
    expect(filterServers(servers, 'private document', none)).toHaveLength(1)

    const connected = filterServers(servers, '', { transports: [], statuses: ['connected'] })
    expect(connected).toHaveLength(13)
    expect(connected[0]?.config.name).toBe('Server 00')
    expect(filterServers(servers, '', { transports: ['stdio'], statuses: [] })).toHaveLength(0)
    expect(paginateMCPItems(connected, 1, 8)).toHaveLength(8)
    expect(paginateMCPItems(connected, 2, 8)).toHaveLength(5)
  })
})
