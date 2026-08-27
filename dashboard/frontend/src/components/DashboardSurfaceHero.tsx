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
  meta?: DashboardSurfaceHeroMeta[]
}

export default function DashboardSurfaceHero({
  compact = false,
  eyebrow = 'Manager',
  title,
  description,
  meta = [],
}: DashboardSurfaceHeroProps) {
  return (
    <header className={`${styles.hero} ${compact ? styles.heroCompact : ''}`}>
      <div className={styles.copy}>
        <div className={styles.topline}>
          <span className={styles.eyebrow}>{eyebrow}</span>
        </div>
        <h1 className={styles.title}>{title}</h1>
        <p className={styles.description}>{description}</p>
        {meta.length > 0 ? (
          <dl className={styles.metaRow}>
            {meta.map((item) => (
              <div key={item.label} className={styles.metaItem}>
                <dt className={styles.metaLabel}>{item.label}</dt>
                <dd className={styles.metaValue}>{item.value}</dd>
              </div>
            ))}
          </dl>
        ) : null}
      </div>
    </header>
  )
}
