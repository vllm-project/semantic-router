import React from 'react'
import DashboardSurfaceHero, {
  type DashboardSurfaceHeroMeta,
} from '../components/DashboardSurfaceHero'
import styles from '../components/DashboardManagerLayout.module.css'

interface FleetSimSurfaceLayoutProps {
  title: string
  description: string
  meta: DashboardSurfaceHeroMeta[]
  children: React.ReactNode
}

export default function FleetSimSurfaceLayout({
  title,
  description,
  meta,
  children,
}: FleetSimSurfaceLayoutProps) {
  return (
    <section className={styles.page}>
      <DashboardSurfaceHero
        eyebrow="Fleet Sim"
        title={title}
        description={description}
        meta={meta}
      />

      <div className={styles.body}>{children}</div>
    </section>
  )
}
