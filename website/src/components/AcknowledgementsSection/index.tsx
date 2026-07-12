import React from 'react'
import Translate from '@docusaurus/Translate'
import { SectionLabel } from '@site/src/components/site/Chrome'
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
    <section className={styles.section}>
      <div className="site-shell-container">
        <div className={styles.header}>
          <div>
            <SectionLabel>
              <Translate id="acknowledgements.label">Ecosystem</Translate>
            </SectionLabel>
            <h2 className={styles.title}>
              <Translate id="acknowledgements.title">Built with open source.</Translate>
            </h2>
          </div>
          <p className={styles.subtitle}>
            <Translate id="acknowledgements.subtitle">
              The libraries and platforms beneath vLLM Semantic Router.
            </Translate>
          </p>
        </div>

        <div className={styles.projectsGrid}>
          {projects.map(project => (
            <a
              key={project.id}
              href={project.url}
              target="_blank"
              rel="noopener noreferrer"
              className={styles.projectCard}
              title={project.name}
              data-project-id={project.id}
            >
              <div className={styles.projectCardTop}>
                <div className={styles.projectLogoWrapper}>
                  <img
                    src={project.logo}
                    alt={project.name}
                    className={styles.projectLogo}
                  />
                </div>
                <span className={styles.projectMeta}>
                  <Translate id="acknowledgements.projectMeta">Dependency</Translate>
                </span>
              </div>
              <div className={styles.projectCopy}>
                <span className={styles.projectName}>{project.name}</span>
                <span className={styles.projectLink}>
                  <Translate id="acknowledgements.projectLink">Open project</Translate>
                </span>
              </div>
            </a>
          ))}
        </div>
      </div>
    </section>
  )
}

export default AcknowledgementsSection
