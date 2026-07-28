import type { ReactNode } from 'react'
import Link from '@docusaurus/Link'
import { useLocation } from '@docusaurus/router'
import Translate from '@docusaurus/Translate'
import clsx from 'clsx'
import React from 'react'
import { normalizeWebsitePath } from '@site/src/components/site/WebsiteMegaNav/navigation'
import styles from './styles.module.css'

export interface ResearchNavItem {
  key: string
  label: string
  to: string
}

export const RESEARCH_NAV_ITEMS: ResearchNavItem[] = [
  { key: 'publications', label: 'Papers & Talks', to: '/publications' },
  { key: 'white-paper', label: 'White Paper', to: '/white-paper' },
  { key: 'vision-paper', label: 'Vision Paper', to: '/vision-paper' },
]

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
        <header className={styles.masthead}>
          <span className={styles.eyebrow}>
            <Translate id="research.layout.eyebrow">Research</Translate>
          </span>
          <h1>{title}</h1>
          {description && <p className={styles.description}>{description}</p>}
        </header>

        <div className={styles.body}>
          <div className={styles.sidebar}>
            <nav className={styles.sidebarNav} aria-label="Research sections">
              {RESEARCH_NAV_ITEMS.map((item) => {
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

            <div className={styles.sidebarCard}>
              <span className={styles.sidebarCardLabel}>
                <Translate id="research.layout.sidebarCard.label">Resources</Translate>
              </span>
              <a
                className={styles.sidebarCardLink}
                href="https://github.com/vllm-project/semantic-router"
                target="_blank"
                rel="noreferrer"
              >
                <Translate id="research.layout.sidebarCard.repo">GitHub Repository</Translate>
                <span aria-hidden="true">↗</span>
              </a>
              <a
                className={styles.sidebarCardLink}
                href="https://huggingface.co/LLM-Semantic-Router"
                target="_blank"
                rel="noreferrer"
              >
                <Translate id="research.layout.sidebarCard.models">Models on Hugging Face</Translate>
                <span aria-hidden="true">↗</span>
              </a>
            </div>
          </div>

          <article className={styles.article}>{children}</article>
        </div>
      </main>
    </div>
  )
}
