import type { ReactNode } from 'react'
import Link from '@docusaurus/Link'
import { useLocation } from '@docusaurus/router'
import clsx from 'clsx'
import React from 'react'
import PageHeader from '@site/src/components/site/PageHeader'
import { normalizeWebsitePath } from '@site/src/components/site/WebsiteMegaNav/navigation'
import styles from './styles.module.css'

export interface BrowseNavItem {
  key: string
  label: string
  to: string
}

export interface BrowseNavGroup {
  key: string
  label: string
  items: BrowseNavItem[]
}

export interface BrowseLayoutProps {
  activeKey: string
  title: ReactNode
  description?: ReactNode
  eyebrow?: ReactNode
  actions?: ReactNode
  sidebarLabel: string
  groups: BrowseNavGroup[]
  children: ReactNode
}

/**
 * Shared browse shell: PageHeader above a left rail and article.
 * Research, Community, and Model Hub all sit in this frame so outer edges
 * and the rail divider line up with Docs.
 */
export default function BrowseLayout({
  activeKey,
  title,
  description,
  eyebrow,
  actions,
  sidebarLabel,
  groups,
  children,
}: BrowseLayoutProps): ReactNode {
  const { pathname, hash } = useLocation()
  const normalizedPathname = normalizeWebsitePath(pathname)

  return (
    <div className={styles.page}>
      <main className={styles.container}>
        <PageHeader
          actions={actions}
          description={description}
          eyebrow={eyebrow}
          title={title}
        />

        <div className={styles.body}>
          <nav className={styles.sidebar} aria-label={sidebarLabel}>
            {groups.map(group => (
              <div key={group.key} className={styles.navGroup}>
                <span className={styles.navGroupLabel}>{group.label}</span>
                {group.items.map((item) => {
                  const itemUrl = new URL(item.to, 'https://vllm-sr.ai')
                  const pathMatches = normalizedPathname === itemUrl.pathname
                  const isHashItem = Boolean(itemUrl.hash)
                  const isActive = isHashItem
                    ? pathMatches && (
                      hash === itemUrl.hash
                      || (!hash && item.key === activeKey)
                    )
                    : item.key === activeKey || pathMatches

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
