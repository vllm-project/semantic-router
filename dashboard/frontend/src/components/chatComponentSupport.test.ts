import { describe, expect, it } from 'vitest'

import type { StoredConversation } from '../hooks'
import type { Message } from './ChatComponentTypes'
import {
  buildFeedbackInsightsHref,
  resolveActiveConversationPreference,
} from './chatComponentSupport'

const conversation = (id: string): StoredConversation<Message[]> => ({
  id,
  createdAt: 1,
  updatedAt: 1,
  payload: [],
})

describe('playground active conversation preference', () => {
  it('restores only an explicitly selected conversation that still exists', () => {
    expect(
      resolveActiveConversationPreference('selected', [
        conversation('newest'),
        conversation('selected'),
      ]),
    ).toBe('selected')
  })

  it('returns the blank starting state when the stored conversation is gone', () => {
    expect(resolveActiveConversationPreference('deleted', [conversation('available')])).toBeNull()
  })
})

describe('playground feedback Insights link', () => {
  it('links to the exact replay record and URL-encodes its id', () => {
    expect(buildFeedbackInsightsHref('/insights/', 'replay/id')).toBe('/insights/replay%2Fid')
    expect(buildFeedbackInsightsHref(undefined, 'replay-id')).toBeUndefined()
  })
})
