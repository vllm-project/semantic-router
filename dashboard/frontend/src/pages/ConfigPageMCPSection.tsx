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
      panelEyebrow="Integrations"
      panelTitle="MCP Control Plane"
      panelDescription="Operate remote MCP servers and reconcile the dashboard tool inventory with router-visible integrations."
      pills={[
        { label: 'MCP Servers', active: true },
        { label: 'Discovered Tools' },
        { label: 'Built-in Registry' },
      ]}
    >
      <div className={styles.sectionPanel}>
        <div className={styles.sectionTableBlock}>
          <MCPConfigPanel embedded />
        </div>
      </div>
    </ConfigPageManagerLayout>
  )
}
