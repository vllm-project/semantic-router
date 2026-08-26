import type { AgentSessionMode } from '../generated/managementApiContract'

export type PlaygroundMode = 'chat' | 'agent' | 'builder'
type AgentPlaygroundMode = Exclude<PlaygroundMode, 'chat'>

export function agentSessionMode(mode: AgentPlaygroundMode): AgentSessionMode {
  return mode === 'builder' ? 'builder' : 'chat'
}

export function playgroundModeForAgentSession(mode: AgentSessionMode): PlaygroundMode {
  return mode === 'builder' ? 'builder' : 'agent'
}
