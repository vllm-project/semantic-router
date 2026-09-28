import React from 'react'
import clsx from 'clsx'
import { useLocation } from '@docusaurus/router'
import Layout from '@theme/Layout'
import BlogSidebar from '@theme/BlogSidebar'
import type { Props } from '@theme/BlogLayout'

function isBlogPostPath(pathname: string): boolean {
  const localeNeutralPath = pathname
    .replace(/^\/zh-Hans(?=\/|$)/i, '')
    .replace(/\/+$/, '')
  const blogPath = localeNeutralPath.replace(/^\/blog/, '')

  return Boolean(blogPath)
    && !blogPath.startsWith('/page/')
    && !blogPath.startsWith('/tags')
    && !blogPath.startsWith('/authors')
}

export default function BlogLayout(props: Props): React.ReactNode {
  const { sidebar, toc, children, ...layoutProps } = props
  const { pathname } = useLocation()
  const isPostPage = isBlogPostPath(pathname)
  const hasSidebar = !isPostPage && sidebar && sidebar.items.length > 0

  return (
    <Layout {...layoutProps}>
      <div
        className={clsx(
          'site-blog-layout',
          isPostPage ? 'site-blog-layout--post' : 'site-blog-layout--listing',
        )}
      >
        <div className="site-blog-layout__inner">
          {hasSidebar && (
            <aside className="site-blog-layout__sidebar">
              <BlogSidebar sidebar={sidebar} />
            </aside>
          )}
          <main className="site-blog-layout__main">{children}</main>
          {toc && <aside className="site-blog-layout__toc">{toc}</aside>}
        </div>
      </div>
    </Layout>
  )
}
