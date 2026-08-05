import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { readFileSync } from 'node:fs'
import { describe, expect, it, vi } from 'vitest'

import { MCPServerDialog } from './MCPServerDialog'
import { MCPToolDetailModal } from './MCPToolDetailModal'
import type { UnifiedTool } from './mcpConfigPanelTypes'

describe('MCP dialog accessibility contracts', () => {
  it('labels the server editor and exposes an intentional initial focus target', () => {
    const markup = renderToStaticMarkup(
      createElement(MCPServerDialog, {
        server: null,
        onClose: vi.fn(),
        onSave: vi.fn(async () => undefined),
        onTest: vi.fn(async () => ({ success: true })),
      }),
    )

    expect(markup).toContain('role="dialog"')
    expect(markup).toContain('aria-modal="true"')
    expect(markup).toMatch(/aria-labelledby="[^"]+"/)
    expect(markup).toContain('aria-label="Close MCP server dialog"')
    expect(markup).toContain('data-dialog-initial-focus="true"')
    expect(markup).toContain('role="radiogroup"')
    expect(markup).toContain('Add argument')
    expect(markup).toContain('No command arguments configured.')
    expect(markup).not.toContain('<textarea')
    expect(markup).not.toContain('one per line')
  })

  it('edits HTTP headers as structured key/value rows', () => {
    const markup = renderToStaticMarkup(
      createElement(MCPServerDialog, {
        server: {
          id: 'docs',
          name: 'Docs server',
          transport: 'streamable-http',
          enabled: true,
          connection: {
            url: 'https://mcp.example.test',
            headers: { Authorization: 'Bearer secret' },
          },
        },
        onClose: vi.fn(),
        onSave: vi.fn(async () => undefined),
        onTest: vi.fn(async () => ({ success: true })),
      }),
    )

    expect(markup).toContain('Request headers')
    expect(markup).toContain('Authorization')
    expect(markup).toContain('Bearer secret')
    expect(markup).toContain('Add header')
    expect(markup).not.toContain('<textarea')
    expect(markup).not.toContain('one per line')
  })

  it('labels tool details with their name and description', () => {
    const tool: UnifiedTool = {
      id: 'mcp-search',
      name: 'search_docs',
      description: 'Search the documentation index.',
      source: 'docs-server',
      sourceType: 'mcp',
      parameters: [],
      rawTool: {} as UnifiedTool['rawTool'],
    }
    const markup = renderToStaticMarkup(
      createElement(MCPToolDetailModal, { tool, onClose: vi.fn() }),
    )

    expect(markup).toContain('role="dialog"')
    expect(markup).toContain('aria-modal="true"')
    expect(markup).toMatch(/aria-labelledby="[^"]+"/)
    expect(markup).toMatch(/aria-describedby="[^"]+"/)
    expect(markup).toContain('aria-label="Close tool details"')
    expect(markup).toContain('Search the documentation index.')
  })

  it('routes server deletion through the shared confirmation dialog', () => {
    const panelSource = readFileSync(new URL('./MCPConfigPanel.tsx', import.meta.url), 'utf8')

    expect(panelSource).toContain('<ConfirmDialog')
    expect(panelSource).not.toMatch(/\b(?:window\.)?confirm\s*\(/)
    expect(panelSource).not.toContain('console.error')
    expect(panelSource).toContain('createLatestMCPRequestRunner')
  })
})
