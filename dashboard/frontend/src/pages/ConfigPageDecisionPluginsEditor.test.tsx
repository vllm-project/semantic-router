import { readFileSync } from 'node:fs'
import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it, vi } from 'vitest'

import ConfigPageDecisionPluginsEditor from './ConfigPageDecisionPluginsEditor'

describe('decision plugin configuration editor', () => {
  it('renders registered plugin fields without a JSON textarea', () => {
    const markup = renderToStaticMarkup(
      createElement(ConfigPageDecisionPluginsEditor, {
        value: [
          {
            type: 'response_cache',
            configuration: { enabled: true, ttl_seconds: 900 },
          },
        ],
        onChange: vi.fn(),
      }),
    )

    expect(markup).toContain('Min Similarity')
    expect(markup).toContain('Enabled')
    expect(markup).not.toContain('Configuration JSON')
    expect(markup).not.toContain('<textarea')
  })

  it('keeps the decision form free of raw JSON configuration input', () => {
    const source = readFileSync(
      new URL('./ConfigPageDecisionsSection.tsx', import.meta.url),
      'utf8',
    )
    expect(source).toContain('<ConfigPageDecisionPluginsEditor')
    expect(source).not.toContain('Configuration JSON')
    expect(source).not.toContain("placeholder='Configuration JSON")
  })

  it('uses the canonical response_cache plugin identifier', () => {
    const source = readFileSync(
      new URL('./ConfigPageDecisionPluginsEditor.tsx', import.meta.url),
      'utf8',
    )
    expect(source).toContain("type: 'response_cache'")
    expect(source).not.toContain("type: 'semantic-cache'")
    expect(source).not.toContain("type: 'semantic_cache'")
  })
})
