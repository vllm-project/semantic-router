import type { ReactNode } from 'react'

import DashboardSurfaceHero, { type DashboardSurfaceHeroMeta } from './DashboardSurfaceHero'
import styles from './DashboardManagerLayout.module.css'

interface DashboardManagerLayoutProps {
  compactHero?: boolean
  eyebrow?: string
  title: string
  description: string
  meta: DashboardSurfaceHeroMeta[]
  children: ReactNode
}

export default function DashboardManagerLayout({
  compactHero = false,
  eyebrow,
  title,
  description,
  meta,
  children,
}: DashboardManagerLayoutProps) {
  return (
    <section className={styles.page}>
      <DashboardSurfaceHero
        compact={compactHero}
        eyebrow={eyebrow}
        title={title}
        description={description}
        meta={meta}
      />
      <div className={styles.body}>{children}</div>
    </section>
  )
}
