import React from 'react'
import styles from './DashboardSurfaceHero.module.css'

export interface DashboardSurfaceHeroMeta {
  label: string
  value: React.ReactNode
}

export interface DashboardSurfaceHeroPill {
  label: React.ReactNode
  active?: boolean
  onClick?: () => void
  disabled?: boolean
}

interface DashboardSurfaceHeroProps {
  eyebrow?: string
  title: string
  description: string
  meta: DashboardSurfaceHeroMeta[]
  panelEyebrow?: string
  panelTitle: string
  panelDescription: string
  pills?: DashboardSurfaceHeroPill[]
  panelFooter?: React.ReactNode
}

export default function DashboardSurfaceHero({
  eyebrow = 'Manager',
  title,
  description,
  meta,
  panelEyebrow = 'Workspace',
  panelTitle,
  panelDescription,
  pills = [],
  panelFooter,
}: DashboardSurfaceHeroProps) {
  return (
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
          {meta.map((item) => (
            <div key={item.label} className={styles.metaCard}>
              <span className={styles.metaLabel}>{item.label}</span>
              <strong className={styles.metaValue}>{item.value}</strong>
            </div>
          ))}
        </div>
      </div>
      <aside className={styles.heroPanel}>
        <div className={styles.panelTop}>
          <div className={styles.logoFrame}>
            <img src="/vllm.png" alt="vLLM" className={styles.panelLogo} />
          </div>
          <div className={styles.panelCopy}>
            <span className={styles.panelEyebrow}>{panelEyebrow}</span>
            <strong className={styles.panelTitle}>{panelTitle}</strong>
            <p className={styles.panelDescription}>{panelDescription}</p>
          </div>
        </div>
        {pills.length > 0 ? (
          <div className={styles.panelPills}>
            {pills.map((pill) =>
              pill.onClick ? (
                <button
                  key={String(pill.label)}
                  type="button"
                  className={`${styles.panelPill} ${pill.active ? styles.panelPillActive : ''} ${styles.panelPillButton}`}
                  onClick={pill.onClick}
                  disabled={pill.disabled}
                >
                  {pill.label}
                </button>
              ) : (
                <span
                  key={String(pill.label)}
                  className={`${styles.panelPill} ${pill.active ? styles.panelPillActive : ''}`}
                >
                  {pill.label}
                </span>
              )
            )}
          </div>
        ) : null}
        {panelFooter ? <div className={styles.panelFooter}>{panelFooter}</div> : null}
      </aside>
    </header>
  )
}
