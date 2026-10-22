import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it, vi } from 'vitest'
import ChatComponentMessages from './ChatComponentMessages'

vi.mock('../contexts/AuthContext', () => ({ useAuth: () => ({ user: null }) }))

describe('incomplete chat messages', () => {
  it('retains the warning next to a partial answer after streaming has ended', () => {
    const markup = renderToStaticMarkup(
      createElement(ChatComponentMessages, {
        canSubmitFeedback: false,
        expandedToolCards: new Set<string>(),
        onToggleToolCard: vi.fn(),
        messages: [
          {
            id: 'partial',
            role: 'assistant',
            content: 'Partial answer.',
            timestamp: new Date(),
            isStreaming: false,
            incomplete: 'Review the backend generation settings and try again.',
          },
        ],
      }),
    )
    expect(markup).toContain('Partial answer.')
    expect(markup).toContain('Incomplete response.')
    expect(markup).toContain('Review the backend generation settings and try again.')
    expect(markup).not.toContain('Generating response')
  })
})

describe('long user message previews', () => {
  const renderMessage = (content: string, role: 'user' | 'assistant' = 'user') =>
    renderToStaticMarkup(
      createElement(ChatComponentMessages, {
        expandedToolCards: new Set<string>(),
        onToggleToolCard: vi.fn(),
        canSubmitFeedback: false,
        messages: [{ id: 'preview', role, content, timestamp: new Date() }],
      }),
    )

  it('bounds line-heavy previews and leaves short messages unchanged', () => {
    const markup = renderMessage(
      Array.from({ length: 20 }, (_, index) => `Line ${index}`).join('\n'),
    )
    expect(markup).toContain('Line 7')
    expect(markup).not.toContain('Line 8')
    expect(markup).toContain('Show more')
    expect(renderMessage('Short message')).not.toContain('Show more')
  })

  it('does not split a surrogate pair at the preview boundary', () => {
    const markup = renderMessage('x'.repeat(1199) + '😀' + ' tail')
    expect(markup).not.toContain('\uD83D')
    expect(markup).toContain('…')
  })
})
