import type { ReactNode } from 'react'
import { MetaItem, PageContainer, PageHeader } from '../ui'
import type { DashboardSurfaceHeroMeta } from './DashboardSurfaceHero'

interface DashboardManagerLayoutProps {
  compactHero?: boolean
  eyebrow?: string
  title: string
  description: string
  meta?: DashboardSurfaceHeroMeta[]
  icon?: ReactNode
  actions?: ReactNode
  children: ReactNode
}

/** Page frame with a compact header; `meta` renders as a quiet facts row. */
export default function DashboardManagerLayout({ title, description, meta, icon, actions, children }: DashboardManagerLayoutProps) {
  return (
    <PageContainer>
      <PageHeader
        title={title}
        description={description}
        icon={icon}
        actions={actions}
        meta={
          meta && meta.length > 0
            ? meta.map((item) => (
                <MetaItem key={item.label} label={item.label}>
                  {item.value}
                </MetaItem>
              ))
            : undefined
        }
      />
      {children}
    </PageContainer>
  )
}
