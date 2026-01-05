import React from 'react'
import Translate from '@docusaurus/Translate'
import styles from './index.module.css'
import acknowledgementsData from './data.json'

interface Project {
  id: string
  name: string
  logo: string
  url: string
}

interface AcknowledgementsData {
  projects: Project[]
}

const typedData: AcknowledgementsData = acknowledgementsData as AcknowledgementsData

const AcknowledgementsSection: React.FC = () => {
  const { projects } = typedData

  return (
    <section className={styles.acknowledgementsSection}>
      <div className="container">
        <div className={styles.acknowledgementsContainer}>
          <h2 className={styles.acknowledgementsTitle}>
            <Translate id="acknowledgements.title">Acknowledgements</Translate>
          </h2>
          <p className={styles.acknowledgementsSubtitle}>
            <Translate id="acknowledgements.subtitle">vLLM Semantic Router is born in open source and built on open source</Translate>
            {' '}
            ❤️
          </p>
          <div className={styles.projectsGrid}>
            {projects.map(project => (
              <a
                key={project.id}
                href={project.url}
                target="_blank"
                rel="noopener noreferrer"
                className={styles.projectCard}
                title={project.name}
              >
                <div className={styles.projectLogoWrapper}>
                  <img
                    src={project.logo}
                    alt={project.name}
                    className={styles.projectLogo}
                  />
                </div>
                <span className={styles.projectName}>{project.name}</span>
              </a>
            ))}
          </div>
        </div>
      </div>
    </section>
  )
}

export default AcknowledgementsSection
