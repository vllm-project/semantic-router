import React from 'react'
import DashboardSurfaceHero from '../components/DashboardSurfaceHero'
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
      <DashboardSurfaceHero
        eyebrow={eyebrow}
        title={title}
        description={description}
        meta={[
          { label: 'Current surface', value: title },
          { label: 'Config area', value: 'Manager' },
          { label: 'Scope', value: 'Live router control' },
        ]}
        panelEyebrow="Workspace"
        panelTitle="Semantic Router Manager"
        panelDescription="Configure the models, decisions, and signals that shape live routing behavior."
        pills={sections.map((section) => ({
          label: section,
          active: section === title,
        }))}
      />

      <div className={styles.body}>{children}</div>
    </section>
  )
}
