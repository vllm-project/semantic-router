import { readFileSync } from 'node:fs'
import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it, vi } from 'vitest'

import {
  McpArgumentsEditor,
  McpEnvironmentEditor,
} from './configPageClassifierStructuredEditors'
import {
  normalizeMcpArguments,
  normalizeMcpCategoryModel,
  normalizeMcpEnvironment,
} from './configPageClassifierStructuredSupport'

describe('classifier structured fields', () => {
  it('normalizes legacy JSON while preserving the MCP payload shape', () => {
    expect(normalizeMcpArguments('["--port", "8080"]')).toEqual(['--port', '8080'])
    expect(normalizeMcpEnvironment('{"API_KEY":"secret"}')).toEqual({ API_KEY: 'secret' })
    expect(
      normalizeMcpCategoryModel({
        enabled: true,
        transport_type: 'stdio',
        threshold: 0.7,
        args: ['--port', '8080'],
        env: { API_KEY: 'secret' },
      }),
    ).toMatchObject({ args: ['--port', '8080'], env: { API_KEY: 'secret' } })
  })

  it('rejects malformed argument and environment schemas before saving', () => {
    expect(() => normalizeMcpArguments('{"port":8080}')).toThrow(/list of text values/i)
    expect(() => normalizeMcpArguments(['--port', ''])).toThrow(/non-empty text/i)
    expect(() => normalizeMcpEnvironment(['API_KEY=secret'])).toThrow(/key\/value/i)
    expect(() => normalizeMcpEnvironment({ API_KEY: 123 })).toThrow(/text values/i)
    expect(() => normalizeMcpEnvironment({ ' API_KEY ': 'one', API_KEY: 'two' })).toThrow(
      /duplicated/i,
    )
  })

  it('renders list and key/value controls without JSON textareas', () => {
    const markup = renderToStaticMarkup(
      createElement(
        'div',
        null,
        createElement(McpArgumentsEditor, { value: ['--port', '8080'], onChange: vi.fn() }),
        createElement(McpEnvironmentEditor, {
          value: { API_KEY: 'secret' },
          onChange: vi.fn(),
        }),
      ),
    )

    expect(markup).toContain('--port')
    expect(markup).toContain('API_KEY')
    expect(markup).not.toContain('textarea')
    expect(markup).not.toContain('JSON')
  })

  it('wires the structured controls into the classifier modal', () => {
    const source = readFileSync(new URL('./ConfigPageClassifierSection.tsx', import.meta.url), 'utf8')
    expect(source).toContain('<McpArgumentsEditor')
    expect(source).toContain('<McpEnvironmentEditor')
    expect(source).toContain('normalizeMcpCategoryModel(data)')
    expect(source).not.toMatch(/Arguments \(JSON\)|Environment Variables \(JSON\)/)
  })
})
