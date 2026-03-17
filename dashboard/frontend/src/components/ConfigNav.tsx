import React from 'react'
import styles from './ConfigNav.module.css'

// New navigation structure aligned with Python CLI config format
export type ConfigSection =
  | 'signals'        // config.yaml: signals (keywords, embeddings, domains, etc.)
  | 'decisions'      // config.yaml: decisions (routing rules)
  | 'models'         // config.yaml: providers.models
  | 'global-config'  // config.yaml: global runtime overrides (cache, prompt guard, tools, etc.)
  | 'mcp'            // MCP servers configuration
  | 'topology'       // Separate page for visualization

interface ConfigNavProps {
  activeSection: ConfigSection
  onSectionChange: (section: ConfigSection) => void
}

const ConfigNav: React.FC<ConfigNavProps> = ({ activeSection, onSectionChange }) => {
  const sections = [
    {
      id: 'global-config' as ConfigSection,
      icon: '⚙️',
      title: 'Global Config',
      description: 'Global runtime overrides, services, stores & model catalog'
    },
    {
      id: 'models' as ConfigSection,
      icon: '🤖',
      title: 'Models',
      description: 'Provider models and endpoints'
    },
    {
      id: 'decisions' as ConfigSection,
      icon: '🔀',
      title: 'Decisions',
      description: 'Routing rules with priorities & plugins'
    },
    {
      id: 'signals' as ConfigSection,
      icon: '📡',
      title: 'Signals',
      description: 'Keywords, embeddings, domains & preferences'
    },
    {
      id: 'mcp' as ConfigSection,
      icon: '🔌',
      title: 'MCP Servers & Tools',
      description: 'MCP servers and all available tools'
    },
    {
      id: 'topology' as ConfigSection,
      icon: '🗺️',
      title: 'Topology',
      description: 'Visualize signal-driven routing flow'
    }
  ]

  return (
    <nav className={styles.nav}>
      <div className={styles.navHeader}>
        <h3 className={styles.navTitle}>Configuration</h3>
      </div>
      <ul className={styles.navList}>
        {sections.map((section) => (
          <li key={section.id}>
            <button
              className={`${styles.navItem} ${activeSection === section.id ? styles.active : ''}`}
              onClick={() => onSectionChange(section.id)}
            >
              <span className={styles.navIcon}>{section.icon}</span>
              <div className={styles.navContent}>
                <span className={styles.navItemTitle}>{section.title}</span>
                <span className={styles.navItemDesc}>{section.description}</span>
              </div>
            </button>
          </li>
        ))}
      </ul>
    </nav>
  )
}

export default ConfigNav
