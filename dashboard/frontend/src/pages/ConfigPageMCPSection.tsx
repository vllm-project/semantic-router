import ConfigPageManagerLayout from './ConfigPageManagerLayout'
import { MCPConfigPanel } from '../components/MCPConfigPanel'
import styles from './ConfigPage.module.css'

export default function ConfigPageMCPSection() {
  return (
    <ConfigPageManagerLayout
      eyebrow="Integrations"
      title="MCP Servers"
      description="Manage Model Context Protocol servers, inspect discovered tools, and keep the dashboard tool inventory aligned with router-side integrations."
      configArea="MCP"
      scope="Tool and server control"
    >
      <div className={styles.sectionPanel}>
        <div className={styles.sectionTableBlock}>
          <MCPConfigPanel embedded />
        </div>
      </div>
    </ConfigPageManagerLayout>
  )
}
