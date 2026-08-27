import { useEffect, useMemo, useState } from 'react'

import AgentManagementPanel, { type AgentManagementTab } from '../components/AgentManagementPanel'
import { useAuth } from '../contexts/AuthContext'
import { canReadAgent, canReadAgentTools } from '../utils/accessControl'
import ConfigPageManagerLayout from './ConfigPageManagerLayout'

const LABELS: Record<AgentManagementTab, string> = {
  skills: 'Skills',
  tools: 'Tools',
  connections: 'Connections',
}

export default function ConfigPageAgentSection() {
  const { user } = useAuth()
  const visibleTabs = useMemo<AgentManagementTab[]>(
    () => [
      ...(canReadAgent(user) ? ['skills' as const] : []),
      ...(canReadAgentTools(user) ? ['tools' as const, 'connections' as const] : []),
    ],
    [user],
  )
  const [activeTab, setActiveTab] = useState<AgentManagementTab>(visibleTabs[0] ?? 'skills')

  useEffect(() => {
    if (!visibleTabs.includes(activeTab) && visibleTabs[0]) setActiveTab(visibleTabs[0])
  }, [activeTab, visibleTabs])

  return (
    <ConfigPageManagerLayout
      eyebrow="Integrations"
      title="vLLM-SR Agent"
      description="Skills and tools for Chat and Builder."
      pills={visibleTabs.map((tab) => ({
        label: LABELS[tab],
        active: tab === activeTab,
        onClick: () => setActiveTab(tab),
      }))}
    >
      <AgentManagementPanel activeTab={activeTab} onTabChange={setActiveTab} />
    </ConfigPageManagerLayout>
  )
}
