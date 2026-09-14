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
