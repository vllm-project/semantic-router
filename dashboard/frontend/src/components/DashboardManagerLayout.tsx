import type { ReactNode } from 'react'

import DashboardSurfaceHero, {
  type DashboardSurfaceHeroMeta,
  type DashboardSurfaceHeroPill,
} from './DashboardSurfaceHero'
import styles from './DashboardManagerLayout.module.css'

interface DashboardManagerLayoutProps {
  compactHero?: boolean
  eyebrow?: string
  title: string
  description: string
  meta: DashboardSurfaceHeroMeta[]
  panelEyebrow?: string
  panelTitle: string
  panelDescription: string
  pills?: DashboardSurfaceHeroPill[]
  panelFooter?: ReactNode
  children: ReactNode
}

export default function DashboardManagerLayout({
  compactHero = false,
  eyebrow,
  title,
  description,
  meta,
  panelEyebrow,
  panelTitle,
  panelDescription,
  pills,
  panelFooter,
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
        panelEyebrow={panelEyebrow}
        panelTitle={panelTitle}
        panelDescription={panelDescription}
        pills={pills}
        panelFooter={panelFooter}
      />
      <div className={styles.body}>{children}</div>
    </section>
  )
}
