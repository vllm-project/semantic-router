import { describe, expect, it } from 'vitest'

import { shouldStreamAgentSessionEvents } from './agentSessionEventPolicy'

describe('Agent session event policy', () => {
  it('keeps custom turn events off the ordinary Chat inference path', () => {
    expect(
      shouldStreamAgentSessionEvents({
        activeSessionId: 'chat-session',
        activeSessionMode: 'chat',
        builderEventsOnly: true,
      }),
    ).toBe(false)
    expect(
      shouldStreamAgentSessionEvents({
        activeSessionId: 'builder-session',
        activeSessionMode: 'builder',
        builderEventsOnly: true,
      }),
    ).toBe(true)
  })

  it('preserves the hook default for other Agent surfaces', () => {
    expect(
      shouldStreamAgentSessionEvents({
        activeSessionId: 'chat-session',
        activeSessionMode: 'chat',
        builderEventsOnly: false,
      }),
    ).toBe(true)
    expect(
      shouldStreamAgentSessionEvents({
        activeSessionId: null,
        activeSessionMode: 'builder',
        builderEventsOnly: false,
      }),
    ).toBe(false)
  })
})
