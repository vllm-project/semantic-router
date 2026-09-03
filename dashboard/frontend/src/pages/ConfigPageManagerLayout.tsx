import React from 'react'
import DashboardManagerLayout from '../components/DashboardManagerLayout'

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
  return (
    <DashboardManagerLayout
      eyebrow={eyebrow}
      title={title}
      description={description}
    >
      {children}
    </DashboardManagerLayout>
  )
}
