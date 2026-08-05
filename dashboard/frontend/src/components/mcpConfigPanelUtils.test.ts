import { describe, expect, it } from 'vitest'

import type { MCPServerState } from '../tools/mcp'
import type { UnifiedTool } from './mcpConfigPanelTypes'
import {
  buildServerConfig,
  buildTestServerConfig,
  filterAndSortServers,
  filterAndSortUnifiedTools,
  getMCPVisibleRange,
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
  url: '',
  headers: {},
  timeout: '30000',
  autoReconnect: true,
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
  })

  it('searches, filters, sorts, and bounds a large tools catalog', () => {
    const tools = Array.from({ length: 48 }, (_, index) =>
      makeTool(index, index % 3 === 0 ? 'mcp' : index % 3 === 1 ? 'frontend' : 'backend'),
    )
    const searched = filterAndSortUnifiedTools(tools, 'private documentation', 'all', 'name-asc')
    expect(searched.map((tool) => tool.id)).toEqual(['backend-17'])

    const mcpTools = filterAndSortUnifiedTools(tools, 'tenant', 'mcp', 'parameters-desc')
    expect(mcpTools).toHaveLength(16)
    expect(mcpTools.every((tool) => tool.sourceType === 'mcp')).toBe(true)
    expect(paginateMCPItems(tools, 2, 12)).toHaveLength(12)
    expect(paginateMCPItems(tools, 2, 12)[0]?.id).toBe('mcp-12')
    expect(getMCPVisibleRange(48, 2, 12)).toEqual({ start: 13, end: 24 })
  })

  it('searches and bounds servers while keeping status and tool-count sorting stable', () => {
    const servers = Array.from({ length: 25 }, (_, index) => makeServer(index))
    expect(filterAndSortServers(servers, 'all', 'private document', 'name-asc')).toHaveLength(1)

    const connected = filterAndSortServers(servers, 'connected', '', 'tools-desc')
    expect(connected).toHaveLength(13)
    expect(connected[0]?.tools?.length).toBeGreaterThanOrEqual(connected[1]?.tools?.length || 0)
    expect(paginateMCPItems(connected, 1, 8)).toHaveLength(8)
    expect(paginateMCPItems(connected, 2, 8)).toHaveLength(5)
  })
})
