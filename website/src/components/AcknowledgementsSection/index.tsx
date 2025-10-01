import React from 'react'
import styles from './index.module.css'
import acknowledgementsData from './data.json'

interface Project {
  id: string
  name: string
  logo: string
  url: string
}

interface AcknowledgementsData {
  title: string
  subtitle: string
  projects: Project[]
}

const typedData: AcknowledgementsData = acknowledgementsData as AcknowledgementsData

const AcknowledgementsSection: React.FC = () => {
  const { title, subtitle, projects } = typedData

  return (
    <section className={styles.acknowledgementsSection}>
      <div className="container">
        <div className={styles.acknowledgementsContainer}>
          <h2 className={styles.acknowledgementsTitle}>
            {title}
          </h2>
          <p className={styles.acknowledgementsSubtitle}>
            {subtitle}
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
