import type { AgentSessionMode } from '../generated/managementApiContract'

interface AgentSessionEventPolicyInput {
  activeSessionId: string | null
  activeSessionMode?: AgentSessionMode
  builderEventsOnly: boolean
}

export function shouldStreamAgentSessionEvents({
  activeSessionId,
  activeSessionMode,
  builderEventsOnly,
}: AgentSessionEventPolicyInput): boolean {
  return Boolean(activeSessionId) && (!builderEventsOnly || activeSessionMode === 'builder')
}
