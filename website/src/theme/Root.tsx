import React from 'react'
import Root from '@theme-original/Root'
import ScrollToTop from '../components/ScrollToTop'
import Head from '@docusaurus/Head'
import { useLocation } from '@docusaurus/router'
import { useEffect, useMemo } from 'react'

function resolveRouteState(pathname: string): { pageKey: string, routeKind: string } {
  const normalized = pathname.replace(/^\/zh-Hans(?=\/|$)/, '') || '/'

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
  const canonicalUrl = `${base}${location.pathname}`.replace(/\/$/, '')
  const routeState = useMemo(() => resolveRouteState(location.pathname), [location.pathname])

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
