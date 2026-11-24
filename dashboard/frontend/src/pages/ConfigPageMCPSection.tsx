import { Blocks } from 'lucide-react'

import { MCPConfigPanel } from '../components/MCPConfigPanel'
import { IconTile } from '../ui'
import ConfigPageManagerLayout from './ConfigPageManagerLayout'

export default function ConfigPageMCPSection() {
  return (
    <MCPConfigPanel
      frame={(actions, content) => (
        <ConfigPageManagerLayout
          title="Integrations"
          description="MCP servers and the tools that decisions can call while they handle a request."
          icon={<IconTile icon={Blocks} tone="slate" size="lg" />}
          actions={actions}
        >
          {content}
        </ConfigPageManagerLayout>
      )}
    />
  )
}
