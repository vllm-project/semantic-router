import React from 'react'
import DashboardManagerLayout from '../components/DashboardManagerLayout'

interface ConfigPageManagerLayoutProps {
  eyebrow?: string
  title: string
  description: string
  configArea?: string
  scope?: string
  children: React.ReactNode
}

export default function ConfigPageManagerLayout({
  eyebrow = 'Manager',
  title,
  description,
  configArea = 'Manager',
  scope = 'Live router control',
  children,
}: ConfigPageManagerLayoutProps) {
  return (
    <DashboardManagerLayout
      eyebrow={eyebrow}
      title={title}
      description={description}
      meta={[
        { label: 'Current surface', value: title },
        { label: 'Config area', value: configArea },
        { label: 'Scope', value: scope },
      ]}
    >
      {children}
    </DashboardManagerLayout>
  )
}
