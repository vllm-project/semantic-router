import type { ReactNode } from 'react'
import Link from '@docusaurus/Link'
import { useLocation } from '@docusaurus/router'
import Translate from '@docusaurus/Translate'
import clsx from 'clsx'
import React from 'react'
import PageHeader from '@site/src/components/site/PageHeader'
import { normalizeWebsitePath } from '@site/src/components/site/WebsiteMegaNav/navigation'
import styles from './styles.module.css'

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
  const { pathname } = useLocation()
  const normalizedPathname = normalizeWebsitePath(pathname)

  return (
    <div className={styles.page}>
      <main className={styles.container}>
        <PageHeader
          description={description}
          eyebrow={<Translate id="research.layout.eyebrow">Research</Translate>}
          title={title}
        />

        <div className={styles.body}>
          <nav className={styles.sidebar} aria-label="Research sections">
            {RESEARCH_NAV_GROUPS.map(group => (
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
