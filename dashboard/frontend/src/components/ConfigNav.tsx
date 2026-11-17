import React from 'react'
import styles from './ConfigNav.module.css'

export type ConfigSection =
  | 'signals' // config.yaml: recipes[].signals (keywords, domains, safety, systemone, etc.)
  | 'projections' // config.yaml: recipes[].projections (partitions, scores, mappings)
  | 'decisions' // config.yaml: recipes[].decisions (routing rules)
  | 'models' // config.yaml: providers.models
  | 'entrypoints-recipes' // config.yaml: top-level entrypoints + recipes
  | 'global-config' // config.yaml: global runtime overrides (stores, services, plugins, model catalog)
  | 'mcp' // MCP servers configuration
  | 'topology' // Separate page for visualization

interface ConfigNavProps {
  activeSection: ConfigSection
  onSectionChange: (section: ConfigSection) => void
}

const ConfigNav: React.FC<ConfigNavProps> = ({ activeSection, onSectionChange }) => {
  const sections = [
    {
      id: 'global-config' as ConfigSection,
      icon: 'GC',
      title: 'Global Config',
      description: 'Global runtime overrides, services, stores & model catalog',
    },
    {
      id: 'decisions' as ConfigSection,
      icon: 'DC',
      title: 'Decisions',
      description: 'Routing rules with priorities & plugins',
    },
    {
      id: 'models' as ConfigSection,
      icon: 'ML',
      title: 'Models',
      description: 'Provider models and endpoints',
    },
    {
      id: 'entrypoints-recipes' as ConfigSection,
      icon: 'MM',
      title: 'Mixture-of-Models',
      description: 'Public MoM models and isolated routing profiles',
    },
    {
      id: 'signals' as ConfigSection,
      icon: 'SG',
      title: 'Signals',
      description: 'Keywords, domains, safety & System One',
    },
    {
      id: 'projections' as ConfigSection,
      icon: 'PJ',
      title: 'Projections',
      description: 'Partitions, scores & derived routing bands',
    },
    {
      id: 'mcp' as ConfigSection,
      icon: 'MP',
      title: 'MCP Servers & Tools',
      description: 'MCP servers and all available tools',
    },
    {
      id: 'topology' as ConfigSection,
      icon: 'TP',
      title: 'Topology',
      description: 'Visualize signal-driven routing flow',
    },
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
