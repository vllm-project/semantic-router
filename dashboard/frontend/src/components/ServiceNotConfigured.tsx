import React from 'react'
import styles from './ServiceNotConfigured.module.css'

export interface ServiceConfig {
  name: string
  envVar: string
  description: string
  docsUrl?: string
  exampleValue?: string
}

interface ServiceNotConfiguredProps {
  service: ServiceConfig
  onRetry?: () => void
}

const ServiceNotConfigured: React.FC<ServiceNotConfiguredProps> = ({
  service,
  onRetry,
}) => {
  return (
    <div className={styles.container}>
      <div className={styles.card}>
        <div className={styles.iconWrapper}>
          <svg
            className={styles.icon}
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <circle cx="12" cy="12" r="10" />
            <line x1="12" y1="8" x2="12" y2="12" />
            <line x1="12" y1="16" x2="12.01" y2="16" />
          </svg>
        </div>

        <h2 className={styles.title}>{service.name} Not Configured</h2>

        <p className={styles.description}>{service.description}</p>

        <div className={styles.configSection}>
          <h3 className={styles.sectionTitle}>Configuration Required</h3>
          <p className={styles.configHint}>
            Set the following environment variable to enable {service.name}:
          </p>
          <code className={styles.envVar}>{service.envVar}</code>
          {service.exampleValue && (
            <div className={styles.example}>
              <span className={styles.exampleLabel}>Example:</span>
              <code className={styles.exampleCode}>
                {service.envVar}={service.exampleValue}
              </code>
            </div>
          )}
        </div>

        <div className={styles.actions}>
          {service.docsUrl && (
            <a
              href={service.docsUrl}
              target="_blank"
              rel="noopener noreferrer"
              className={styles.docsLink}
            >
              <svg
                className={styles.linkIcon}
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
              >
                <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6" />
                <polyline points="15 3 21 3 21 9" />
                <line x1="10" y1="14" x2="21" y2="3" />
              </svg>
              View Documentation
            </a>
          )}
          {onRetry && (
            <button onClick={onRetry} className={styles.retryButton}>
              <svg
                className={styles.retryIcon}
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
              >
                <polyline points="23 4 23 10 17 10" />
                <path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10" />
              </svg>
              Retry Connection
            </button>
          )}
        </div>
      </div>
    </div>
  )
}

export default ServiceNotConfigured
