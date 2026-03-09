import React from 'react'
import styles from './ConfigPageManagerLayout.module.css'

interface ConfigPageManagerLayoutProps {
  eyebrow?: string
  title: string
  description: string
  children: React.ReactNode
}

export default function ConfigPageManagerLayout({
  eyebrow = 'Manager',
  title,
  description,
  children,
}: ConfigPageManagerLayoutProps) {
  const sections = ['Models', 'Decisions', 'Signals']

  return (
    <section className={styles.page}>
      <header className={styles.hero}>
        <div className={styles.heroGlow} aria-hidden="true" />
        <div className={styles.copy}>
          <div className={styles.topline}>
            <div className={styles.brandBadge}>
              <img src="/vllm.png" alt="vLLM" className={styles.brandLogo} />
              <span>vLLM Semantic Router</span>
            </div>
            <span className={styles.eyebrow}>{eyebrow}</span>
          </div>
          <h1 className={styles.title}>{title}</h1>
          <p className={styles.description}>{description}</p>
          <div className={styles.metaRow}>
            <div className={styles.metaCard}>
              <span className={styles.metaLabel}>Current surface</span>
              <strong className={styles.metaValue}>{title}</strong>
            </div>
            <div className={styles.metaCard}>
              <span className={styles.metaLabel}>Config area</span>
              <strong className={styles.metaValue}>Manager</strong>
            </div>
            <div className={styles.metaCard}>
              <span className={styles.metaLabel}>Scope</span>
              <strong className={styles.metaValue}>Live router control</strong>
            </div>
          </div>
        </div>
        <aside className={styles.heroPanel}>
          <div className={styles.panelTop}>
            <div className={styles.logoFrame}>
              <img src="/vllm.png" alt="vLLM" className={styles.panelLogo} />
            </div>
            <div className={styles.panelCopy}>
              <span className={styles.panelEyebrow}>Workspace</span>
              <strong className={styles.panelTitle}>Semantic Router Manager</strong>
              <p className={styles.panelDescription}>
                Configure the models, decisions, and signals that shape live routing behavior.
              </p>
            </div>
          </div>
          <div className={styles.panelPills}>
            {sections.map((section) => (
              <span
                key={section}
                className={`${styles.panelPill} ${section === title ? styles.panelPillActive : ''}`}
              >
                {section}
              </span>
            ))}
          </div>
        </aside>
      </header>

      <div className={styles.body}>{children}</div>
    </section>
  )
}
