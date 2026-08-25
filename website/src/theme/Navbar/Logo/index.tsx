import type { ReactNode } from 'react'
import Link from '@docusaurus/Link'
import { useLocation } from '@docusaurus/router'
import { useThemeConfig } from '@docusaurus/theme-common'
import useBaseUrl from '@docusaurus/useBaseUrl'
import Logo from '@theme/Logo'
import React from 'react'

/**
 * Navbar brand.
 *
 * Everywhere except the landing page this is the stock `Logo`, which swaps
 * `logo.src` / `logo.srcDark` with the reader's color mode.
 *
 * The landing page is the exception: it is a dark-committed design, so
 * shell.css keeps the chrome dark in *both* color modes. The stock component
 * cannot serve that, and not for a CSS reason — `ThemedComponent` renders only
 * the variant matching the current color mode into the DOM, so in light mode
 * the white mark does not exist to be revealed. Hiding the dark-ink one there
 * leaves the brand slot empty, which is what happened.
 *
 * So on the home route the white mark is rendered directly, independent of
 * color mode.
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
