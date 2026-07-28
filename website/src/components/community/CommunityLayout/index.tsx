import type { ReactNode } from 'react'
import Link from '@docusaurus/Link'
import { useLocation } from '@docusaurus/router'
import Translate from '@docusaurus/Translate'
import clsx from 'clsx'
import React from 'react'
import { normalizeWebsitePath } from '@site/src/components/site/WebsiteMegaNav/navigation'
import styles from './styles.module.css'

export interface CommunityNavItem {
  key: string
  label: string
  to: string
}

export const COMMUNITY_NAV_ITEMS: CommunityNavItem[] = [
  { key: 'team', label: 'Team', to: '/community/team' },
  { key: 'steering-committee', label: 'Steering Committee', to: '/community/steering-committee' },
  { key: 'governance', label: 'Governance', to: '/community/governance' },
  { key: 'work-groups', label: 'Working Groups', to: '/community/work-groups' },
  { key: 'contributing', label: 'Contributing', to: '/community/contributing' },
  { key: 'leaderboard', label: 'Leaderboard', to: '/community/contributors' },
  { key: 'code-of-conduct', label: 'Code of Conduct', to: '/community/code-of-conduct' },
]

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
        <header className={styles.masthead}>
          <span className={styles.eyebrow}>
            <Translate id="community.layout.eyebrow">Community</Translate>
          </span>
          <h1>{title}</h1>
          {description && <p className={styles.description}>{description}</p>}
        </header>

        <div className={styles.body}>
          <nav className={styles.sidebar} aria-label="Community sections">
            {COMMUNITY_NAV_ITEMS.map((item) => {
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
          </nav>

          <article className={styles.article}>{children}</article>
        </div>
      </main>
    </div>
  )
}
