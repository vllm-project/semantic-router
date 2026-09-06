import type { ReactNode } from 'react'
import Link from '@docusaurus/Link'
import { useLocation } from '@docusaurus/router'
import Translate from '@docusaurus/Translate'
import clsx from 'clsx'
import React from 'react'
import PageHeader from '@site/src/components/site/PageHeader'
import { normalizeWebsitePath } from '@site/src/components/site/WebsiteMegaNav/navigation'
import styles from './styles.module.css'

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
  const { pathname } = useLocation()
  const normalizedPathname = normalizeWebsitePath(pathname)

  return (
    <div className={styles.page}>
      <main className={styles.container}>
        <PageHeader
          description={description}
          eyebrow={<Translate id="community.layout.eyebrow">Community</Translate>}
          title={title}
        />

        <div className={styles.body}>
          <nav className={styles.sidebar} aria-label="Community sections">
            {COMMUNITY_NAV_GROUPS.map(group => (
              <div key={group.key} className={styles.navGroup}>
                <span className={styles.navGroupLabel}>{group.label}</span>
                {group.items.map((item) => {
                  const isActive = item.key === activeKey || normalizedPathname === item.to

                  return (
                    <Link
                      key={item.key}
                      className={clsx(styles.navLink, {
                        [styles.navLinkActive]: isActive,
                      })}
                      to={item.to}
                      aria-current={isActive ? 'page' : undefined}
                    >
                      {item.label}
                    </Link>
                  )
                })}
              </div>
            ))}
          </nav>

          <article className={styles.article}>{children}</article>
        </div>
      </main>
    </div>
  )
}
