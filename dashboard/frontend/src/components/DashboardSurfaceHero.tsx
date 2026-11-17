import React from 'react'
import { MetaItem, PageHeader } from '../ui'

export interface DashboardSurfaceHeroMeta {
  label: string
  value: React.ReactNode
}

interface DashboardSurfaceHeroProps {
  compact?: boolean
  eyebrow?: string
  title: string
  description: string
  meta?: DashboardSurfaceHeroMeta[]
}

/** Compatibility wrapper: renders the standard page header. */
export default function DashboardSurfaceHero({ title, description, meta = [] }: DashboardSurfaceHeroProps) {
  return (
    <PageHeader
      title={title}
      description={description}
      meta={
        meta.length > 0
          ? meta.map((item) => (
              <MetaItem key={item.label} label={item.label}>
                {item.value}
              </MetaItem>
            ))
          : undefined
      }
    />
  )
}
