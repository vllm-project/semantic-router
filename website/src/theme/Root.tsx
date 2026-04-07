import React, { useEffect, useMemo } from 'react'
import Root from '@theme-original/Root'
import Head from '@docusaurus/Head'
import { useLocation } from '@docusaurus/router'
import ScrollToTop from '../components/ScrollToTop'

function normalizePath(pathname: string): string {
  const normalized = pathname.replace(/\/+$/, '')
  return normalized === '' ? '/' : normalized
}

function stripLocalePrefix(pathname: string): string {
  return normalizePath(pathname.replace(/^\/zh-Hans(?=\/|$)/, ''))
}

function toAbsoluteUrl(base: string, pathname: string): string {
  return pathname === '/' ? base : `${base}${pathname}`
}

function resolveRouteState(pathname: string): { pageKey: string, routeKind: string } {
  const normalized = stripLocalePrefix(pathname)

  if (normalized === '/') {
    return { pageKey: 'home', routeKind: 'home' }
  }

  if (normalized.startsWith('/docs')) {
    return { pageKey: 'docs', routeKind: 'docs' }
  }

  if (normalized.startsWith('/blog')) {
    return { pageKey: 'blog', routeKind: 'blog' }
  }

  if (normalized.startsWith('/publications')) {
    return { pageKey: 'publications', routeKind: 'page' }
  }

  if (normalized.startsWith('/white-paper')) {
    return { pageKey: 'white-paper', routeKind: 'page' }
  }

  if (normalized.startsWith('/community')) {
    return { pageKey: 'community', routeKind: 'page' }
  }

  return { pageKey: 'other', routeKind: 'page' }
}

export default function RootWrapper(props: React.ComponentProps<typeof Root>): React.ReactElement {
  const location = useLocation()
  const base = 'https://vllm-semantic-router.com'
  const pathname = normalizePath(location.pathname || '/')
  const localeNeutralPath = stripLocalePrefix(pathname)
  const canonicalUrl = toAbsoluteUrl(base, pathname)
  const englishUrl = toAbsoluteUrl(base, localeNeutralPath)
  const chineseUrl = toAbsoluteUrl(base, localeNeutralPath === '/' ? '/zh-Hans' : `/zh-Hans${localeNeutralPath}`)
  const routeState = useMemo(() => resolveRouteState(pathname), [pathname])

  useEffect(() => {
    document.documentElement.dataset.routeKind = routeState.routeKind
    document.documentElement.dataset.pageKey = routeState.pageKey
    document.body.dataset.routeKind = routeState.routeKind
    document.body.dataset.pageKey = routeState.pageKey

    return () => {
      delete document.documentElement.dataset.routeKind
      delete document.documentElement.dataset.pageKey
      delete document.body.dataset.routeKind
      delete document.body.dataset.pageKey
    }
  }, [routeState])

  return (
    <>
      <Head>
        <link rel="canonical" href={canonicalUrl} />
        <link rel="alternate" hrefLang="en" href={englishUrl} />
        <link rel="alternate" hrefLang="zh-Hans" href={chineseUrl} />
        <link rel="alternate" hrefLang="x-default" href={englishUrl} />
        <meta property="og:url" content={canonicalUrl} />
        <meta name="twitter:url" content={canonicalUrl} />
      </Head>
      <div className={`site-root site-root--${routeState.routeKind} site-page--${routeState.pageKey}`}>
        <Root {...props} />
      </div>
      <ScrollToTop />
    </>
  )
}
