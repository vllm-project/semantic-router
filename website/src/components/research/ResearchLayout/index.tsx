import type { ReactNode } from 'react'
import Translate from '@docusaurus/Translate'
import React from 'react'
import BrowseLayout from '@site/src/components/site/BrowseLayout'

export interface ResearchNavItem {
  key: string
  label: string
  to: string
}

export interface ResearchNavGroup {
  key: string
  label: string
  items: ResearchNavItem[]
}

/* Grouped like the docs sidebar, so the rail reads the same on both. */
export const RESEARCH_NAV_GROUPS: ResearchNavGroup[] = [
  {
    key: 'publications',
    label: 'Publications',
    items: [
      { key: 'publications', label: 'Papers & Talks', to: '/publications' },
      { key: 'white-paper', label: 'White Paper', to: '/white-paper' },
      { key: 'vision-paper', label: 'Vision Paper', to: '/vision-paper' },
    ],
  },
]

export const RESEARCH_NAV_ITEMS: ResearchNavItem[] = RESEARCH_NAV_GROUPS.flatMap(
  group => group.items,
)

export interface ResearchLayoutProps {
  activeKey: string
  title: ReactNode
  description?: ReactNode
  children: ReactNode
}

export default function ResearchLayout({
  activeKey,
  title,
  description,
  children,
}: ResearchLayoutProps): ReactNode {
  return (
    <BrowseLayout
      activeKey={activeKey}
      description={description}
      eyebrow={<Translate id="research.layout.eyebrow">Research</Translate>}
      groups={RESEARCH_NAV_GROUPS}
      sidebarLabel="Research sections"
      title={title}
    >
      {children}
    </BrowseLayout>
  )
}
