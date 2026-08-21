import type { ReactNode } from 'react'

import type { DashboardSurfaceHeroPill } from '../components/DashboardSurfaceHero'
import styles from './ConfigPageManagerLayout.module.css'

interface ConfigPageManagerLayoutProps {
  eyebrow?: string
  title: string
  description: string
  configArea?: string
  scope?: string
  panelEyebrow?: string
  panelTitle?: string
  panelDescription?: string
  pills?: DashboardSurfaceHeroPill[]
  children: ReactNode
}

export default function ConfigPageManagerLayout({
  eyebrow = 'Build',
  title,
  description,
  pills = [],
  children,
}: ConfigPageManagerLayoutProps) {
  const hasPanelNavigation = pills.length > 0

  return (
    <section className={styles.page}>
      <header className={styles.header}>
        <div className={styles.headerGrid} aria-hidden="true" />
        <div className={styles.copy}>
          <div className={styles.topline}>
            <span>{eyebrow}</span>
            <div className={styles.brand}>
              <img src="/vllm.png" alt="" />
              <span>Semantic Router</span>
            </div>
          </div>
          <h1>{title}</h1>
          <p>{description}</p>
        </div>
        {hasPanelNavigation ? (
          <aside className={styles.surfacePulse}>
            <div className={styles.pills} aria-label={`${title} views`}>
              {pills.map((pill) =>
                pill.onClick ? (
                  <button
                    key={String(pill.label)}
                    type="button"
                    className={pill.active ? styles.activePill : ''}
                    onClick={pill.onClick}
                    disabled={pill.disabled}
                  >
                    {pill.label}
                  </button>
                ) : (
                  <span key={String(pill.label)} className={pill.active ? styles.activePill : ''}>
                    {pill.label}
                  </span>
                ),
              )}
            </div>
          </aside>
        ) : null}
      </header>
      <div className={styles.body}>{children}</div>
    </section>
  )
}
