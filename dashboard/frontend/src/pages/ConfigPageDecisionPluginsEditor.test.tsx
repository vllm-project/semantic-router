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
            type: 'semantic_cache',
            configuration: { enabled: true, similarity_threshold: 0.91 },
          },
        ],
        onChange: vi.fn(),
      }),
    )

    expect(markup).toContain('Similarity Threshold')
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
})
