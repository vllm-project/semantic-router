import type {
  AgentProfile,
  AgentSkill,
  AgentToolDefinition,
  AgentToolSource,
} from '../generated/managementApiContract'
import type { AgentManagementTab } from './AgentManagementPanel'
import type { AgentEditableResource } from './AgentResourceEditor'

export type AgentResource = AgentProfile | AgentSkill | AgentToolDefinition | AgentToolSource

export function resourceId(tab: AgentManagementTab, resource: AgentResource): string {
  return tab === 'tools'
    ? (resource as AgentToolDefinition).name
    : (resource as AgentEditableResource).id
}

export function resourceName(resource: AgentResource): string {
  return resource.name
}
