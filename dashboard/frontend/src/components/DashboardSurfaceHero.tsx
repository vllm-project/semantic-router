import React from 'react'
import styles from './DashboardSurfaceHero.module.css'

export interface DashboardSurfaceHeroMeta {
  label: string
  value: React.ReactNode
}

interface DashboardSurfaceHeroProps {
  compact?: boolean
  eyebrow?: string
  title: string
  description: string
  meta: DashboardSurfaceHeroMeta[]
}

export default function DashboardSurfaceHero({
  compact = false,
  eyebrow = 'Manager',
  title,
  description,
  meta,
}: DashboardSurfaceHeroProps) {
  return (
    <header className={`${styles.hero} ${compact ? styles.heroCompact : ''}`}>
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
          {meta.map((item) => (
            <div key={item.label} className={styles.metaCard}>
              <span className={styles.metaLabel}>{item.label}</span>
              <strong className={styles.metaValue}>{item.value}</strong>
            </div>
          ))}
        </div>
      </div>
    </header>
  )
}
