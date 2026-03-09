import React from 'react'
import clsx from 'clsx'
import Layout from '@theme/Layout'
import BlogSidebar from '@theme/BlogSidebar'
import type { Props } from '@theme/BlogLayout'
import { PillLink, SectionLabel } from '@site/src/components/site/Chrome'

export default function BlogLayout(props: Props): React.ReactNode {
  const { sidebar, toc, children, ...layoutProps } = props
  const hasSidebar = sidebar && sidebar.items.length > 0
  const title = typeof layoutProps.title === 'string'
    ? layoutProps.title
    : 'Journal'
  const description = typeof layoutProps.description === 'string'
    ? layoutProps.description
    : 'Release notes, field reports, and research commentary from the vLLM Semantic Router project.'

  return (
    <Layout {...layoutProps}>
      <header className="site-blog-masthead">
        <div className="site-shell-container">
          <SectionLabel>Blog</SectionLabel>
          <div className="site-blog-masthead__row">
            <div className="site-blog-masthead__copy">
              <h1>{title}</h1>
              <p>{description}</p>
            </div>
            <div className="site-blog-masthead__actions">
              <PillLink to="/docs/intro">Documentation</PillLink>
              <PillLink href="https://github.com/vllm-project/semantic-router" muted target="_blank" rel="noreferrer">
                GitHub
              </PillLink>
            </div>
          </div>
        </div>
      </header>

      <div className="site-blog-shell">
        <div className="site-shell-container site-blog-shell__inner">
          <div className="row">
            <BlogSidebar sidebar={sidebar} />
            <main
              className={clsx('col', {
                'col--7': hasSidebar,
                'col--9 col--offset-1': !hasSidebar,
              })}
            >
              {children}
            </main>
            {toc && <div className="col col--2">{toc}</div>}
          </div>
        </div>
      </div>
    </Layout>
  )
}
