import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it, vi } from 'vitest'

import type { MCPServerState } from '../tools/mcp'
import { TooltipProvider } from '../ui'
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

const render = (element: ReturnType<typeof createElement>) =>
  renderToStaticMarkup(createElement(TooltipProvider, null, element))

function serversSection(servers: MCPServerState[], isReadonly: boolean) {
  return render(
    createElement(MCPServersSection, {
      servers,
      filteredServers: servers,
      serverFilter: 'all',
      serverSearch: '',
      serverSort: 'name-asc',
      isReadonly,
      actionLoading: null,
      onServerFilterChange: vi.fn(),
      onServerSearchChange: vi.fn(),
      onServerSortChange: vi.fn(),
      onOpenServer: vi.fn(),
      onToggleConnection: vi.fn(),
      onToggleEnabled: vi.fn(),
      onEditServer: vi.fn(),
      onDeleteServer: vi.fn(),
      onAddServer: vi.fn(),
      onRefresh: vi.fn(),
    }),
  )
}

describe('MCP catalog sections', () => {
  it('renders a bounded tools page instead of the full catalog', () => {
    const tools = Array.from({ length: 31 }, (_, index) => makeTool(index))
    const markup = render(
      createElement(MCPAvailableToolsSection, {
        allAvailableTools: tools,
        filteredTools: tools,
        toolSearch: '',
        toolSort: 'name-asc',
        toolSourceFilter: 'all',
        onSearchChange: vi.fn(),
        onSelectTool: vi.fn(),
        onSortChange: vi.fn(),
        onSourceFilterChange: vi.fn(),
      }),
    )

    expect(markup.match(/aria-label="View details for Tool/g)).toHaveLength(12)
    expect(markup).toContain('1–12')
    expect(markup).toContain('31</span> tools')
    expect(markup).toContain('12 per page')
    expect(markup).toContain('aria-label="Next tools page"')
  })

  it('renders one bounded server page with its status', () => {
    const markup = serversSection(
      Array.from({ length: 20 }, (_, index) => makeServer(index)),
      true,
    )

    expect(markup).toContain('Server 00')
    expect(markup).toContain('Server 07')
    expect(markup).not.toContain('Server 08')
    expect(markup).toContain('8 per page')
    expect(markup).toContain('Connected')
    expect(markup).toContain('Not connected')
  })

  it('offers adding a server only to people who may manage them', () => {
    expect(serversSection([], true)).toContain('No MCP servers are configured.')
    expect(serversSection([], true)).not.toContain('>Add server<')
    expect(serversSection([], false)).toContain('Add server')
  })
})
