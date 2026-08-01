import React from 'react'
import Link from '@docusaurus/Link'
import { translate } from '@docusaurus/Translate'
import type { Props } from '@theme/BlogListPaginator'

export default function BlogListPaginator({
  metadata,
}: Props): React.ReactNode {
  const { previousPage, nextPage } = metadata
  const newerLabel = translate({
    id: 'theme.blog.paginator.newerPosts',
    message: 'Newer',
  })
  const olderLabel = translate({
    id: 'theme.blog.paginator.olderPosts',
    message: 'Older',
  })
  const pageLabel = translate({
    id: 'theme.blog.paginator.pageStatus',
    message: `Page ${metadata.page} of ${metadata.totalPages}`,
  })

  return (
    <nav
      aria-label={translate({
        id: 'theme.blog.paginator.navAriaLabel',
        message: 'Blog list page navigation',
      })}
      className="site-blog-pagination"
    >
      <div className="site-blog-pagination__slot site-blog-pagination__slot--newer">
        {previousPage && (
          <Link
            aria-label={newerLabel}
            className="site-blog-pagination__link"
            to={previousPage}
          >
            <span aria-hidden="true">←</span>
            <span>{newerLabel}</span>
          </Link>
        )}
      </div>
      <span className="site-blog-pagination__status">{pageLabel}</span>
      <div className="site-blog-pagination__slot site-blog-pagination__slot--older">
        {nextPage && (
          <Link
            aria-label={olderLabel}
            className="site-blog-pagination__link"
            to={nextPage}
          >
            <span>{olderLabel}</span>
            <span aria-hidden="true">→</span>
          </Link>
        )}
      </div>
    </nav>
  )
}
