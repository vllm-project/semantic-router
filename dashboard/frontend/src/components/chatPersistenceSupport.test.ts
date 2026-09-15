import { describe, expect, it } from 'vitest'

import { sanitizeMessagesForPersistence } from './chatPersistenceSupport'
import type { Message } from './ChatComponentTypes'

describe('chatPersistenceSupport', () => {
  it('keeps display text but removes inline image bytes from browser persistence', () => {
    const message: Message = {
      id: 'image-message',
      role: 'user',
      content: 'Inspect this diagram.',
      requestContent: [
        { type: 'text', text: 'Inspect this diagram.' },
        { type: 'image_url', image_url: { url: 'data:image/png;base64,AA==' } },
      ],
      images: [{ src: 'data:image/png;base64,AA==', alt: 'diagram' }],
      playgroundAttachments: [
        {
          id: 'image',
          fileName: 'diagram.png',
          sizeBytes: 1,
          content: 'data:image/png;base64,AA==',
          kind: 'image',
        },
      ],
      timestamp: new Date(),
    }

    const persisted = sanitizeMessagesForPersistence([message])

    expect(persisted[0].content).toBe('Inspect this diagram.')
    expect(persisted[0].requestContent).toBeUndefined()
    expect(persisted[0].images).toBeUndefined()
    expect(persisted[0].playgroundAttachments).toBeUndefined()
    expect(JSON.stringify(persisted)).not.toContain('data:image')
  })
})
