import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { readFileSync } from 'node:fs'
import { describe, expect, it, vi } from 'vitest'

import type { MCPServerConfig } from '../tools/mcp'
import { MCPServerFormBody } from './MCPServerDialog'
import { MCP_REDACTED_ARGUMENTS, type ServerFormValues } from './mcpConfigPanelUtils'

const values = (overrides: Partial<ServerFormValues> = {}): ServerFormValues => ({
  name: '',
  description: '',
  transport: 'stdio',
  enabled: true,
  command: '',
  args: [],
  preserveArguments: false,
  url: '',
  headers: {},
  timeout: '30000',
  ...overrides,
})

const form = (server: MCPServerConfig | null, formValues: ServerFormValues) =>
  renderToStaticMarkup(
    createElement(MCPServerFormBody, {
      server,
      values: formValues,
      busy: false,
      onChange: vi.fn(),
    }),
  )

describe('MCP server editor', () => {
  it('labels the fields and offers structured arguments with an initial focus target', () => {
    const markup = form(null, values())

    expect(markup).toContain('data-dialog-initial-focus="true"')
    expect(markup).toContain('role="radiogroup"')
    expect(markup).toContain('Add argument')
    expect(markup).toContain('No command arguments configured.')
    expect(markup).not.toContain('<textarea')
  })

  it('edits HTTP headers as structured key/value rows', () => {
    const server: MCPServerConfig = {
      id: 'docs',
      name: 'Docs server',
      transport: 'streamable-http',
      enabled: true,
      connection: { url: 'https://mcp.example.test', headers: { Authorization: 'Bearer secret' } },
    }
    const markup = form(
      server,
      values({
        transport: 'streamable-http',
        url: 'https://mcp.example.test',
        headers: { Authorization: 'Bearer secret' },
      }),
    )

    expect(markup).toContain('Request headers')
    expect(markup).toContain('Authorization')
    expect(markup).toContain('Bearer secret')
    expect(markup).toContain('Add header')
    expect(markup).not.toContain('<textarea')
  })

  it('shows hidden stdio arguments as an explicit preserve-or-replace state', () => {
    const server: MCPServerConfig = {
      id: 'filesystem',
      name: 'Filesystem server',
      transport: 'stdio',
      enabled: true,
      connection: { command: 'npx', args: [MCP_REDACTED_ARGUMENTS] },
    }
    const markup = form(
      server,
      values({ name: 'Filesystem server', command: 'npx', preserveArguments: true }),
    )

    expect(markup).toContain('Configured arguments are hidden and will be preserved.')
    expect(markup).toContain('Replace or clear arguments')
    expect(markup).toContain(
      'Transport cannot be changed after creation. Create a new server instead.',
    )
    expect(markup).not.toContain(MCP_REDACTED_ARGUMENTS)
  })

  it('routes server deletion through the shared confirmation dialog', () => {
    const panelSource = readFileSync(new URL('./MCPConfigPanel.tsx', import.meta.url), 'utf8')

    expect(panelSource).toContain('<ConfirmDialog')
    expect(panelSource).not.toMatch(/\b(?:window\.)?confirm\s*\(/)
    expect(panelSource).not.toContain('console.error')
    expect(panelSource).toContain('createLatestMCPRequestRunner')
    expect(panelSource).toContain(
      'authLoading || readonlyLoading || serverReadonly || !canManageServers',
    )
    expect(panelSource).not.toContain('runtimeConfigWritable')
    expect(panelSource).toContain('server-wide read-only policy disables MCP server changes')
  })

  it('labels the tool details dialog and its close button', () => {
    const source = readFileSync(new URL('./MCPToolDetailModal.tsx', import.meta.url), 'utf8')
    expect(source).toContain('aria-label="Close tool details"')
    expect(source).toContain('<DialogTitle')
    expect(source).toContain('<DialogDescription')
  })
})
