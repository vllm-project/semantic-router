import React from 'react'
import styles from './ConfigNav.module.css'

export type ConfigSection =
  | 'models'
  | 'prompt-guard'
  | 'similarity-cache'
  | 'intelligent-routing'
  | 'tools-selection'
  | 'observability'

interface ConfigNavProps {
  activeSection: ConfigSection
  onSectionChange: (section: ConfigSection) => void
}

const ConfigNav: React.FC<ConfigNavProps> = ({ activeSection, onSectionChange }) => {
  const sections = [
    {
      id: 'models' as ConfigSection,
      icon: '🔌',
      title: 'Models & Endpoints',
      description: 'Model configurations and backend endpoints'
    },
    {
      id: 'prompt-guard' as ConfigSection,
      icon: '🛡️',
      title: 'Prompt Guard',
      description: 'PII and jailbreak detection'
    },
    {
      id: 'similarity-cache' as ConfigSection,
      icon: '⚡',
      title: 'Similarity Cache',
      description: 'Semantic caching configuration'
    },
    {
      id: 'intelligent-routing' as ConfigSection,
      icon: '📊',
      title: 'Intelligent Routing',
      description: 'Categories and reasoning configuration'
    },
    {
      id: 'tools-selection' as ConfigSection,
      icon: '🔧',
      title: 'Tools Selection',
      description: 'Tool auto-selection settings'
    },
    {
      id: 'observability' as ConfigSection,
      icon: '📈',
      title: 'Observability',
      description: 'Metrics and monitoring'
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
