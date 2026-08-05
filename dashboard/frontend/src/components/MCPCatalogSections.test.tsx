import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it, vi } from 'vitest'

import type { MCPServerState } from '../tools/mcp'
import { MCPAvailableToolsSection } from './MCPAvailableToolsSection'
import { MCPServersSection } from './MCPServersSection'
import type { UnifiedTool } from './mcpConfigPanelTypes'

function makeTool(index: number): UnifiedTool {
  return {
    id: `tool-${index}`,
    name: `Tool ${index}`,
    description: `Description ${index}`,
    source: 'Catalog',
    sourceType: index % 2 === 0 ? 'mcp' : 'backend',
    parameters: [],
    rawTool: {} as UnifiedTool['rawTool'],
  }
}

function makeServer(index: number): MCPServerState {
  return {
    config: {
      id: `server-${index}`,
      name: `Server ${String(index).padStart(2, '0')}`,
      transport: 'streamable-http',
      connection: { url: `https://server-${index}.example.test` },
      enabled: true,
    },
    status: index % 2 === 0 ? 'connected' : 'disconnected',
  }
}

describe('MCP enterprise catalog sections', () => {
  it('renders an accessible, bounded tools page instead of the full catalog', () => {
    const tools = Array.from({ length: 31 }, (_, index) => makeTool(index))
    const markup = renderToStaticMarkup(
      createElement(MCPAvailableToolsSection, {
        allAvailableTools: tools,
        filteredTools: tools,
        toolSearch: '',
        toolSort: 'name-asc',
        toolSourceFilter: 'all',
        toolsSectionExpanded: true,
        onSearchChange: vi.fn(),
        onSelectTool: vi.fn(),
        onSortChange: vi.fn(),
        onSourceFilterChange: vi.fn(),
        onToggleExpanded: vi.fn(),
      }),
    )

    expect(markup).toContain('aria-expanded="true"')
    expect(markup.match(/aria-label="View details for Tool/g)).toHaveLength(12)
    expect(markup).toContain('1–12 of 31 tools')
    expect(markup).toContain('Client view · 12 per page')
    expect(markup).toContain('aria-label="Next tools page"')
  })

  it('renders only one bounded server page and keeps management controls disabled in read-only mode', () => {
    const servers = Array.from({ length: 20 }, (_, index) => makeServer(index))
    const markup = renderToStaticMarkup(
      createElement(MCPServersSection, {
        actionLoading: null,
        builtInExpanded: false,
        expandedServers: new Set<string>(),
        filteredServers: servers,
        isReadonly: true,
        registryTools: [],
        serverFilter: 'all',
        serverSearch: '',
        serverSort: 'name-asc',
        servers,
        serversSectionExpanded: true,
        toolsDbExpanded: false,
        toolsDbLoading: false,
        toolsDbTools: [],
        onDeleteServer: vi.fn(),
        onEditServer: vi.fn(),
        onSelectTool: vi.fn(),
        onServerFilterChange: vi.fn(),
        onServerSearchChange: vi.fn(),
        onServerSortChange: vi.fn(),
        onToggleBuiltInExpanded: vi.fn(),
        onToggleConnection: vi.fn(),
        onToggleServerExpand: vi.fn(),
        onToggleServersSection: vi.fn(),
        onToggleToolsDbExpanded: vi.fn(),
      }),
    )

    expect(markup).toContain('aria-expanded="true"')
    expect(markup.match(/aria-label="Edit Server/g)).toHaveLength(8)
    expect(markup.match(/aria-label="Delete Server/g)).toHaveLength(8)
    expect(markup).toContain('1–8 of 20 servers')
    expect(markup).toContain('Client view · 8 per page')
    expect(markup).toMatch(/disabled=""[^>]*aria-label="Edit Server 00"/)
  })
})
