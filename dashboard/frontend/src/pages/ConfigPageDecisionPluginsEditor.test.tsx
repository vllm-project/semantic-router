import { readFileSync } from 'node:fs'
import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it, vi } from 'vitest'

import { PLUGIN_TYPES } from '../generated/routerConfigContract'
import { TooltipProvider } from '../ui'
import ConfigPageDecisionPluginsEditor from './ConfigPageDecisionPluginsEditor'
import { PLUGINS } from './configPageDecisionPresentation'

describe('decision plugin configuration editor', () => {
  it('renders registered plugin fields from the schema, not a JSON textarea', () => {
    const markup = renderToStaticMarkup(
      createElement(
        TooltipProvider,
        null,
        createElement(ConfigPageDecisionPluginsEditor, {
          value: [{ type: 'response_cache', configuration: { enabled: true, ttl_seconds: 900 } }],
          onChange: vi.fn(),
        }),
      ),
    )
    expect(markup).toContain('Response Cache')
    expect(markup).toContain('Enabled')
    expect(markup).toContain('TTL Seconds')
    expect(markup).toContain('more settings at their defaults')
    expect(markup).not.toContain('<textarea')
  })

  it('edits an unregistered plugin type as JSON and keeps it', () => {
    const markup = renderToStaticMarkup(
      createElement(
        TooltipProvider,
        null,
        createElement(ConfigPageDecisionPluginsEditor, {
          value: [{ type: 'custom_extension', configuration: { anything: 1 } }],
          onChange: vi.fn(),
        }),
      ),
    )
    expect(markup).toContain('Not in the contract')
    expect(markup).toContain('custom_extension')
  })

  it('offers every plugin type of the Router contract under its canonical identifier', () => {
    expect(PLUGINS.map((plugin) => plugin.type)).toEqual(PLUGIN_TYPES)
    expect(PLUGIN_TYPES).toContain('response_cache')
    const source = readFileSync(
      new URL('./ConfigPageDecisionPluginsEditor.tsx', import.meta.url),
      'utf8',
    )
    expect(source).not.toContain('semantic-cache')
    expect(source).not.toContain('semantic_cache')
    expect(source).not.toContain('Configuration JSON')
  })
})
