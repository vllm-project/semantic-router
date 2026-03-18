import React from 'react'
import { useNavigate } from 'react-router-dom'
import DashboardSurfaceHero, { type DashboardSurfaceHeroMeta } from '../components/DashboardSurfaceHero'
import { FLEET_SIM_NAV_ITEMS } from '../utils/fleetSimApi'
import styles from './ConfigPageManagerLayout.module.css'

interface FleetSimSurfaceLayoutProps {
  title: string
  description: string
  currentPath: string
  meta: DashboardSurfaceHeroMeta[]
  panelFooter?: React.ReactNode
  children: React.ReactNode
}

export default function FleetSimSurfaceLayout({
  title,
  description,
  currentPath,
  meta,
  panelFooter,
  children,
}: FleetSimSurfaceLayoutProps) {
  const navigate = useNavigate()

  return (
    <section className={styles.page}>
      <DashboardSurfaceHero
        eyebrow="Fleet Sim"
        title={title}
        description={description}
        meta={meta}
        panelEyebrow="Planning Workspace"
        panelTitle="Capacity planning"
        panelDescription="Review traffic inputs, reusable fleets, and saved planning runs in one operator view."
        pills={FLEET_SIM_NAV_ITEMS.map((item) => ({
          label: item.label,
          active: item.to === currentPath,
          onClick: () => navigate(item.to),
        }))}
        panelFooter={panelFooter}
      />

      <div className={styles.body}>{children}</div>
    </section>
  )
}
