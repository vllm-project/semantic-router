import type { ReactNode } from 'react'
import clsx from 'clsx'
import React from 'react'
import styles from './styles.module.css'

export interface PageHeaderProps {
  /** Section the page belongs to — Blog, Research, Community. */
  eyebrow?: ReactNode
  title: ReactNode
  description?: ReactNode
  /** Trailing controls (search, filters) laid out beside the title on desktop. */
  actions?: ReactNode
  className?: string
}

/**
 * The one page title on the site.
 *
 * Blog, Research and Community each grew their own masthead — 3.25rem/600,
 * 4.25rem/400 and 2em/400 respectively — so three pages one nav click apart
 * looked like three different sites. Section identity now lives in the
 * eyebrow, and the title itself is the same component everywhere.
 */
export default function PageHeader({
  eyebrow,
  title,
  description,
  actions,
  className,
}: PageHeaderProps): ReactNode {
  return (
    <header className={clsx('site-page-header', styles.header, className)}>
      <div className={styles.text}>
        {eyebrow && <span className={styles.eyebrow}>{eyebrow}</span>}
        <h1 className={styles.title}>{title}</h1>
        {description && <p className={styles.description}>{description}</p>}
      </div>
      {actions && <div className={styles.actions}>{actions}</div>}
    </header>
  )
}
