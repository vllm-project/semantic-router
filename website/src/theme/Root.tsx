import React from 'react'
import Root from '@theme-original/Root'
import ScrollToTop from '../components/ScrollToTop'
import Head from '@docusaurus/Head'
import { useLocation } from '@docusaurus/router'

export default function RootWrapper(props: React.ComponentProps<typeof Root>): React.ReactElement {
  const location = useLocation()
  const base = 'https://vllm-semantic-router.com'
  const canonicalUrl = `${base}${location.pathname}`.replace(/\/$/, '')
  return (
    <>
      <Head>
        <link rel="canonical" href={canonicalUrl} />
        <meta property="og:url" content={canonicalUrl} />
        <meta name="twitter:url" content={canonicalUrl} />
      </Head>
      <Root {...props} />
      <ScrollToTop />
    </>
  )
}
