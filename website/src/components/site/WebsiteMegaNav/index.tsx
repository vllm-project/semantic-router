import type { ReactNode } from 'react'
import Link from '@docusaurus/Link'
import { useLocation } from '@docusaurus/router'
import clsx from 'clsx'
import React from 'react'
import {
  isWebsitePrimaryNavItemActive,
  WEBSITE_PRIMARY_NAV_ITEMS,
} from './navigation'
import styles from './index.module.css'

export default function WebsiteMegaNav(): ReactNode {
  const { pathname } = useLocation()

  return (
    <nav className={styles.root} aria-label="Primary navigation">
      {WEBSITE_PRIMARY_NAV_ITEMS.map(item => (
        <Link
          key={item.key}
          className={clsx(styles.link, {
            [styles.linkActive]: isWebsitePrimaryNavItemActive(item, pathname),
          })}
          to={item.to}
        >
          {item.label}
        </Link>
      ))}
    </nav>
  )
}
