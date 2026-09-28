import React from 'react'
import Link from '@docusaurus/Link'
import { useBlogPost } from '@docusaurus/plugin-content-blog/client'
import BlogPostItemContainer from '@theme/BlogPostItem/Container'
import BlogPostItemHeader from '@theme/BlogPostItem/Header'
import BlogPostItemContent from '@theme/BlogPostItem/Content'
import BlogPostItemFooter from '@theme/BlogPostItem/Footer'
import type { Props } from '@theme/BlogPostItem'

function formatDate(date: string): string {
  return new Intl.DateTimeFormat('en-US', {
    day: 'numeric',
    month: 'long',
    timeZone: 'UTC',
    year: 'numeric',
  }).format(new Date(date))
}

export default function BlogPostItem({
  children,
  className,
}: Props): React.ReactNode {
  const { isBlogPostPage, metadata } = useBlogPost()

  if (!isBlogPostPage) {
    return (
      <BlogPostItemContainer className={className}>
        <BlogPostItemHeader />
        <BlogPostItemContent>{children}</BlogPostItemContent>
        <BlogPostItemFooter />
      </BlogPostItemContainer>
    )
  }

  const {
    authors,
    date,
    readingTime,
    tags,
    title,
  } = metadata
  const namedAuthors = authors.filter(author => Boolean(author.name))

  return (
    <BlogPostItemContainer className={className}>
      <header className="site-blog-post__header">
        <Link className="site-blog-post__breadcrumb" to="/blog">
          Blog
        </Link>
        <h1>{title}</h1>
        <div className="site-blog-post__meta">
          <time dateTime={date}>{formatDate(date)}</time>
          {typeof readingTime !== 'undefined' && (
            <span>{`${Math.ceil(readingTime)} min read`}</span>
          )}
        </div>
        {namedAuthors.length > 0 && (
          <ul aria-label="Authors" className="site-blog-post__authors">
            {namedAuthors.map((author, index) => (
              <li className="site-blog-post__author" key={author.key ?? author.name ?? index}>
                {author.imageURL && (
                  <img
                    alt=""
                    className="site-blog-post__author-avatar"
                    loading="lazy"
                    src={author.imageURL}
                  />
                )}
                <div className="site-blog-post__author-details">
                  {author.url
                    ? (
                        <Link to={author.url}>{author.name}</Link>
                      )
                    : (
                        <span>{author.name}</span>
                      )}
                  {author.title && <small>{author.title}</small>}
                </div>
              </li>
            ))}
          </ul>
        )}
        {tags.length > 0 && (
          <ul aria-label="Tags" className="site-blog-post__tags">
            {tags.map(tag => (
              <li key={tag.permalink}>
                <Link to={tag.permalink}>{`#${tag.label}`}</Link>
              </li>
            ))}
          </ul>
        )}
      </header>
      <BlogPostItemContent className="site-blog-post__content">
        {children}
      </BlogPostItemContent>
    </BlogPostItemContainer>
  )
}
