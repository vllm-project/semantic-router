import type { ReactNode } from 'react'
import Translate, { translate } from '@docusaurus/Translate'
import React from 'react'
import BrowseLayout, { type BrowseNavGroup } from '@site/src/components/site/BrowseLayout'

export interface CommunityNavItem {
  key: string
  labelId: string
  label: string
  to: string
}

export interface CommunityNavGroup {
  key: string
  labelId: string
  label: string
  items: CommunityNavItem[]
}

/* Grouped like the docs sidebar. The labels and their order are unchanged
 * from the workgroups reset; the rule falls between the fourth and fifth.
 * `key` and `to` stay literal so destinations and active state cannot drift;
 * only the visible text is translated. */
export const COMMUNITY_NAV_GROUPS: CommunityNavGroup[] = [
  {
    key: 'people',
    labelId: 'community.nav.group.people',
    label: 'People',
    items: [
      { key: 'team', labelId: 'community.nav.item.team', label: 'Open Source Team', to: '/community/team' },
      { key: 'steering-committee', labelId: 'community.nav.item.steeringCommittee', label: 'Steering Committee', to: '/community/steering-committee' },
      { key: 'work-groups', labelId: 'community.nav.item.workGroups', label: 'Working Groups', to: '/community/work-groups' },
      { key: 'leaderboard', labelId: 'community.nav.item.leaderboard', label: 'Leaderboard', to: '/community/contributors' },
    ],
  },
  {
    key: 'how-we-work',
    labelId: 'community.nav.group.howWeWork',
    label: 'How we work',
    items: [
      { key: 'governance', labelId: 'community.nav.item.governance', label: 'Governance', to: '/community/governance' },
      { key: 'contributing', labelId: 'community.nav.item.contributing', label: 'Contributing', to: '/community/contributing' },
      { key: 'code-of-conduct', labelId: 'community.nav.item.codeOfConduct', label: 'Code of Conduct', to: '/community/code-of-conduct' },
    ],
  },
]

export const COMMUNITY_NAV_ITEMS: CommunityNavItem[] = COMMUNITY_NAV_GROUPS.flatMap(
  group => group.items,
)

/* The browse rail renders plain strings, so the labels go through the imperative
 * `translate` API; the data above keeps its stable keys and destinations. */
function localizedNavGroups(): BrowseNavGroup[] {
  return COMMUNITY_NAV_GROUPS.map(group => ({
    key: group.key,
    label: translate({ id: group.labelId, message: group.label }),
    items: group.items.map(item => ({
      key: item.key,
      label: translate({ id: item.labelId, message: item.label }),
      to: item.to,
    })),
  }))
}

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
      groups={localizedNavGroups()}
      sidebarLabel={translate({
        id: 'community.layout.sidebarLabel',
        message: 'Community sections',
      })}
      title={title}
    >
      {children}
    </BrowseLayout>
  )
}
