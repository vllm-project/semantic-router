import { describe, expect, it } from 'vitest'

import { agentSessionMode, playgroundModeForAgentSession } from './playgroundModes'

describe('Playground mode boundaries', () => {
  it('keeps direct Chat outside Agent sessions while mapping Agent and Builder explicitly', () => {
    expect(agentSessionMode('agent')).toBe('chat')
    expect(agentSessionMode('builder')).toBe('builder')
    expect(playgroundModeForAgentSession('chat')).toBe('agent')
    expect(playgroundModeForAgentSession('builder')).toBe('builder')
  })
})
