import type { AgentSessionMode } from '../generated/managementApiContract'

export type PlaygroundMode = 'chat' | 'builder'

export function agentSessionMode(mode: PlaygroundMode): AgentSessionMode {
  return mode
}

export function playgroundModeForAgentSession(mode: AgentSessionMode): PlaygroundMode {
  return mode
}
