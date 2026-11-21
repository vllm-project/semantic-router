import type { ReactNode } from 'react'
import Translate, { translate } from '@docusaurus/Translate'
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
 * from the workgroups reset; the rule falls between the fourth and fifth.
 *
 * Every label is a literal `translate()` call: Docusaurus extracts translation
 * IDs by statically evaluating the argument, so a computed id never reaches
 * `write-translations`. `key` and `to` stay literal so destinations and the
 * active-page state cannot drift; only the visible text is translated. */
export const COMMUNITY_NAV_GROUPS: CommunityNavGroup[] = [
  {
    key: 'people',
    label: translate({ id: 'community.nav.group.people', message: 'People' }),
    items: [
      { key: 'team', label: translate({ id: 'community.nav.item.team', message: 'Open Source Team' }), to: '/community/team' },
      { key: 'steering-committee', label: translate({ id: 'community.nav.item.steeringCommittee', message: 'Steering Committee' }), to: '/community/steering-committee' },
      { key: 'work-groups', label: translate({ id: 'community.nav.item.workGroups', message: 'Working Groups' }), to: '/community/work-groups' },
      { key: 'leaderboard', label: translate({ id: 'community.nav.item.leaderboard', message: 'Leaderboard' }), to: '/community/contributors' },
    ],
  },
  {
    key: 'how-we-work',
    label: translate({ id: 'community.nav.group.howWeWork', message: 'How we work' }),
    items: [
      { key: 'governance', label: translate({ id: 'community.nav.item.governance', message: 'Governance' }), to: '/community/governance' },
      { key: 'contributing', label: translate({ id: 'community.nav.item.contributing', message: 'Contributing' }), to: '/community/contributing' },
      { key: 'code-of-conduct', label: translate({ id: 'community.nav.item.codeOfConduct', message: 'Code of Conduct' }), to: '/community/code-of-conduct' },
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
