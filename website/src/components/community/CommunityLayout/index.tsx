import type { ReactNode } from 'react'
import Translate from '@docusaurus/Translate'
import React from 'react'
import BrowseLayout from '@site/src/components/site/BrowseLayout'

export interface CommunityNavItem {
  key: string
  label: string
  to: string
}

export interface CommunityNavGroup {
  key: string
  label: string
  items: CommunityNavItem[]
}

/* Grouped like the docs sidebar. The labels and their order are unchanged
 * from the workgroups reset; the rule falls between the fourth and fifth. */
export const COMMUNITY_NAV_GROUPS: CommunityNavGroup[] = [
  {
    key: 'people',
    label: 'People',
    items: [
      { key: 'team', label: 'Open Source Team', to: '/community/team' },
      { key: 'steering-committee', label: 'Steering Committee', to: '/community/steering-committee' },
      { key: 'work-groups', label: 'Working Groups', to: '/community/work-groups' },
      { key: 'leaderboard', label: 'Leaderboard', to: '/community/contributors' },
    ],
  },
  {
    key: 'how-we-work',
    label: 'How we work',
    items: [
      { key: 'governance', label: 'Governance', to: '/community/governance' },
      { key: 'contributing', label: 'Contributing', to: '/community/contributing' },
      { key: 'code-of-conduct', label: 'Code of Conduct', to: '/community/code-of-conduct' },
    ],
  },
]

export const COMMUNITY_NAV_ITEMS: CommunityNavItem[] = COMMUNITY_NAV_GROUPS.flatMap(
  group => group.items,
)

export interface CommunityLayoutProps {
  activeKey: string
  title: ReactNode
  description?: ReactNode
  children: ReactNode
}

export default function CommunityLayout({
  activeKey,
  title,
  description,
  children,
}: CommunityLayoutProps): ReactNode {
  return (
    <BrowseLayout
      activeKey={activeKey}
      description={description}
      eyebrow={<Translate id="community.layout.eyebrow">Community</Translate>}
      groups={COMMUNITY_NAV_GROUPS}
      sidebarLabel="Community sections"
      title={title}
    >
      {children}
    </BrowseLayout>
  )
}
