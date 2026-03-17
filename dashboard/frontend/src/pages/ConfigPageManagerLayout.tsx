import React from 'react'
import DashboardSurfaceHero from '../components/DashboardSurfaceHero'
import styles from './ConfigPageManagerLayout.module.css'
import type { DashboardSurfaceHeroPill } from '../components/DashboardSurfaceHero'

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
  children: React.ReactNode
}

export default function ConfigPageManagerLayout({
  eyebrow = 'Manager',
  title,
  description,
  configArea = 'Manager',
  scope = 'Live router control',
  panelEyebrow = 'Workspace',
  panelTitle = 'Semantic Router Manager',
  panelDescription = 'Configure the models, decisions, and signals that shape live routing behavior.',
  pills,
  children,
}: ConfigPageManagerLayoutProps) {
  const defaultPills: DashboardSurfaceHeroPill[] = ['Models', 'Decisions', 'Signals'].map((section) => ({
    label: section,
    active: section === title,
  }))

  return (
    <section className={styles.page}>
      <DashboardSurfaceHero
        eyebrow={eyebrow}
        title={title}
        description={description}
        meta={[
          { label: 'Current surface', value: title },
          { label: 'Config area', value: configArea },
          { label: 'Scope', value: scope },
        ]}
        panelEyebrow={panelEyebrow}
        panelTitle={panelTitle}
        panelDescription={panelDescription}
        pills={pills ?? defaultPills}
      />

      <div className={styles.body}>{children}</div>
    </section>
  )
}
