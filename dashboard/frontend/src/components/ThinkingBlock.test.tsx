import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it, vi } from 'vitest'
import ThinkingBlock from './ThinkingBlock'

vi.mock('thinking-orbs', () => ({
  ThinkingOrb: ({ paused, 'aria-label': label }: { paused?: boolean; 'aria-label'?: string }) =>
    createElement('canvas', { 'data-paused': String(paused), 'aria-label': label }),
}))

describe('ThinkingBlock completion status', () => {
  it('stops the animation and labels finished reasoning without implying ongoing composition', () => {
    const markup = renderToStaticMarkup(
      createElement(ThinkingBlock, {
        content: 'Reasoning content.',
        isStreaming: false,
      }),
    )
    expect(markup).toContain('data-paused="true"')
    expect(markup).toContain('aria-label="Reasoning finished"')
    expect(markup).toContain('>Reasoning</span>')
    expect(markup).not.toContain('Composing')
    expect(markup).toContain('Reasoning content.')
  })

  it('keeps an active indicator while the response is streaming', () => {
    const markup = renderToStaticMarkup(
      createElement(ThinkingBlock, {
        content: 'Reasoning in progress.',
        isStreaming: true,
      }),
    )
    expect(markup).toContain('data-paused="false"')
    expect(markup).toContain('aria-label="Thinking"')
    expect(markup).toContain('>Thinking</span>')
  })
})
