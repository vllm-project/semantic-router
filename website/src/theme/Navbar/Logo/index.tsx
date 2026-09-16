import type { ReactNode } from 'react'
import Link from '@docusaurus/Link'
import { useLocation } from '@docusaurus/router'
import { useThemeConfig } from '@docusaurus/theme-common'
import useBaseUrl from '@docusaurus/useBaseUrl'
import Logo from '@theme/Logo'
import React from 'react'

/**
 * Navbar brand.
 * Stock Logo swaps src/srcDark with color mode. Homepage is dark-committed,
 * so white mark renders directly (ThemedComponent only renders current mode variant).
 */
function isHomeRoute(pathname: string): boolean {
  const localeNeutralPath = pathname
    .replace(/^\/zh-Hans(?=\/|$)/i, '')
    .replace(/\/+$/, '')

  return localeNeutralPath === ''
}

export default function NavbarLogo(): ReactNode {
  const { pathname } = useLocation()
  const { navbar: { logo, title } } = useThemeConfig()

  const logoHref = useBaseUrl(logo?.href || '/')
  const darkSrc = useBaseUrl(logo?.srcDark || logo?.src || '')

  if (logo && isHomeRoute(pathname)) {
    return (
      <Link className="navbar__brand" to={logoHref}>
        <div className="navbar__logo">
          <img alt={logo.alt ?? ''} src={darkSrc} />
        </div>
        {title != null && <b className="navbar__title text--truncate">{title}</b>}
      </Link>
    )
  }

  return (
    <Logo
      className="navbar__brand"
      imageClassName="navbar__logo"
      titleClassName="navbar__title text--truncate"
    />
  )
}
