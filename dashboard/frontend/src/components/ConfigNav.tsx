import React from 'react'
import styles from './ConfigNav.module.css'

// New navigation structure aligned with Python CLI config format
export type ConfigSection =
  | 'signals' // config.yaml: signals (keywords, embeddings, domains, etc.)
  | 'projections' // config.yaml: routing.projections (partitions, scores, mappings)
  | 'decisions' // config.yaml: decisions (routing rules)
  | 'models' // Router-managed Models
  | 'entrypoints-recipes' // Router-managed entrypoints + recipes
  | 'agent' // Router-native Agent profiles, skills, tools, and connections

interface ConfigNavProps {
  activeSection: ConfigSection
  onSectionChange: (section: ConfigSection) => void
}

const ConfigNav: React.FC<ConfigNavProps> = ({ activeSection, onSectionChange }) => {
  const sections = [
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
      description: 'Keywords, embeddings, domains & preferences',
    },
    {
      id: 'projections' as ConfigSection,
      icon: 'PJ',
      title: 'Projections',
      description: 'Partitions, scores & derived routing bands',
    },
    {
      id: 'agent' as ConfigSection,
      icon: 'AG',
      title: 'vLLM-SR Agent',
      description: 'Skills, tools, and connections',
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
