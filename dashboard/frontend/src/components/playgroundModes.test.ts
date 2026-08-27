import { describe, expect, it } from 'vitest'

import { agentSessionMode, playgroundModeForAgentSession } from './playgroundModes'

describe('Playground mode boundaries', () => {
  it('maps Chat and Builder onto the same session kernel without a third Agent mode', () => {
    expect(agentSessionMode('chat')).toBe('chat')
    expect(agentSessionMode('builder')).toBe('builder')
    expect(playgroundModeForAgentSession('chat')).toBe('chat')
    expect(playgroundModeForAgentSession('builder')).toBe('builder')
  })
})
