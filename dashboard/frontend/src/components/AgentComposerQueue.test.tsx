import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'

import AgentComposerQueue from './AgentComposerQueue'

describe('Agent Composer queue', () => {
  it('keeps queued turns compact, legible, and individually removable', () => {
    const markup = renderToStaticMarkup(
      createElement(AgentComposerQueue, {
        paused: true,
        turns: [
          {
            id: 'follow-up',
            input: 'Compare the chosen route with the fastest option',
            attachments: [],
          },
          {
            id: 'image-check',
            input: 'Now inspect the image',
            attachments: [
              {
                id: 'image',
                fileName: 'diagram.png',
                sizeBytes: 128,
                content: 'data:image/png;base64,AA==',
                kind: 'image',
                mediaType: 'image/png',
              },
            ],
          },
        ],
        onRemove: () => undefined,
        onResume: () => undefined,
      }),
    )

    expect(markup).toContain('aria-label="Queued messages"')
    expect(markup).toContain('Queued')
    expect(markup).toContain('>Resume<')
    expect(markup).toContain('Compare the chosen route with the fastest option')
    expect(markup).toContain('1 file')
    expect(markup).toContain(
      'aria-label="Remove queued message: Compare the chosen route with the fastest option"',
    )
  })

  it('stays out of the DOM when there is no queued work', () => {
    const markup = renderToStaticMarkup(
      createElement(AgentComposerQueue, {
        paused: false,
        turns: [],
        onRemove: () => undefined,
        onResume: () => undefined,
      }),
    )

    expect(markup).toBe('')
  })
})
