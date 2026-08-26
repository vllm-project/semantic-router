import type { AgentManagementTab } from './AgentManagementPanel'
import type { AgentResource } from './AgentManagementResourceSupport'

export function resourcesForAgentTab(
  activeTab: AgentManagementTab,
  resourceTab: AgentManagementTab | null,
  resources: AgentResource[],
): AgentResource[] {
  return activeTab === resourceTab ? resources : []
}
