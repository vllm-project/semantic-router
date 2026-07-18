import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'

import ClawRoomTranscript from './ClawRoomTranscript'
import type { RoomMessage } from './clawRoomChatSupport'

describe('ClawRoom transcript scale', () => {
  it('renders a bounded recent window with an explicit earlier-history control', () => {
    const messages: RoomMessage[] = Array.from({ length: 105 }, (_, index) => ({
      id: `message-${index}`,
      roomId: 'room-1',
      teamId: 'team-1',
      senderType: 'user',
      senderName: 'Operator',
      content:
        index === 0 ? 'oldest-hidden' : index === 104 ? 'newest-visible' : `content-${index}`,
      createdAt: '2026-07-12T00:00:00Z',
    }))

    const markup = renderToStaticMarkup(
      createElement(ClawRoomTranscript, {
        selectedRoomId: 'room-1',
        messages,
        streamingMessages: new Map(),
        streamingParticipants: new Map(),
        streamingEntries: [],
        streamingToolTraces: new Map(),
        resolveSenderVisual: () => ({ displayName: 'Operator', roleLabel: 'USER' }),
      }),
    )

    expect(markup).toContain('Load earlier messages')
    expect(markup).toContain('5 earlier messages')
    expect(markup).not.toContain('oldest-hidden')
    expect(markup).toContain('newest-visible')
  })
})
