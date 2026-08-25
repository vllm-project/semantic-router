import React, { useMemo } from 'react'
import Link from '@docusaurus/Link'
import useDocusaurusContext from '@docusaurus/useDocusaurusContext'
import {
  HtmlClassNameProvider,
  PageMetadata,
  ThemeClassNames,
} from '@docusaurus/theme-common'
import SearchMetadata from '@theme/SearchMetadata'
import BlogLayout from '@theme/BlogLayout'
import PageHeader from '@site/src/components/site/PageHeader'
import BlogListPaginator from '@theme/BlogListPaginator'
import BlogListPageStructuredData from '@theme/BlogListPage/StructuredData'
import type { Props } from '@theme/BlogListPage'
import type { BlogSearchIndexEntry } from '../../plugins/blogSearchIndex'

interface BlogCardProps {
  featured?: boolean
  entry: BlogSearchIndexEntry
}

interface TagSummary {
  count: number
  label: string
  permalink: string
}

function formatDate(date: string): string {
  return new Intl.DateTimeFormat('en-US', {
    day: 'numeric',
    month: 'short',
    timeZone: 'UTC',
    year: 'numeric',
  }).format(new Date(date))
}

function readingTimeLabel(readingTime: number | undefined): string | null {
  if (typeof readingTime === 'undefined') {
    return null
  }

  return `${Math.ceil(readingTime)} min read`
}

/**
 * Normalizes a post from `props.items` into a plain card shape shared by the
 * featured and grid cards.
 */
function toEntry(item: Props['items'][number]): BlogSearchIndexEntry {
  const { frontMatter, metadata } = item.content

  return {
    permalink: metadata.permalink,
    title: metadata.title,
    date: metadata.date,
    description: metadata.description ?? '',
    image: typeof frontMatter.image === 'string' ? frontMatter.image : undefined,
    readingTime: metadata.readingTime,
    tags: metadata.tags.map(tag => tag.label),
  }
}

function BlogPostImage({
  alt,
  image,
}: {
  alt: string
  image: string | undefined
}): React.ReactNode {
  if (!image) {
    return (
      <div className="site-blog-card__image-fallback" aria-hidden="true">
        <span>vLLM</span>
        <strong>Semantic Router</strong>
      </div>
    )
  }

  return (
    <div className="site-blog-card__image-frame">
      <span
        aria-hidden="true"
        className="site-blog-card__image-backdrop"
        style={{ backgroundImage: `url(${image})` }}
      />
      <img
        alt={alt}
        className="site-blog-card__image"
        loading="lazy"
        src={image}
      />
    </div>
  )
}

function BlogCard({ featured = false, entry }: BlogCardProps): React.ReactNode {
  const { date, description, image, permalink, readingTime, tags, title } = entry
  const readTime = readingTimeLabel(readingTime)
  const chip = featured ? 'Featured' : tags[0]

  return (
    <article
      className={
        featured
          ? 'site-blog-card site-blog-card--featured'
          : 'site-blog-card'
      }
    >
      <Link
        aria-label={`Read ${title}`}
        className="site-blog-card__media"
        to={permalink}
      >
        <BlogPostImage alt="" image={image} />
      </Link>
      <div className="site-blog-card__body">
        {chip && <span className="site-blog-card__eyebrow">{chip}</span>}
        <h2 className="site-blog-card__title">
          <Link to={permalink}>{title}</Link>
        </h2>
        <div className="site-blog-card__meta">
          <time dateTime={date}>{formatDate(date)}</time>
          {readTime && (
            <>
              <span aria-hidden="true">·</span>
              <span>{readTime}</span>
            </>
          )}
        </div>
        {featured && description && (
          <p className="site-blog-card__description">{description}</p>
        )}
      </div>
    </article>
  )
}

function BlogListPageMetadata({ metadata }: Props): React.ReactNode {
  const {
    siteConfig: { title: siteTitle },
  } = useDocusaurusContext()
  const { blogDescription, blogTitle, permalink } = metadata
  const title = permalink === '/' ? siteTitle : blogTitle

  return (
    <>
      <PageMetadata title={title} description={blogDescription} />
      <SearchMetadata tag="blog_posts_list" />
    </>
  )
}

export default function BlogListPage(props: Props): React.ReactNode {
  const { items, metadata } = props

  const pageEntries = useMemo(() => items.map(toEntry), [items])

  const tagSummaries = useMemo(() => {
    const counts = new Map<string, TagSummary>()

    items.forEach(({ content: BlogPostContent }) => {
      BlogPostContent.metadata.tags.forEach((tag) => {
        const current = counts.get(tag.permalink)
        counts.set(tag.permalink, {
          count: (current?.count ?? 0) + 1,
          label: tag.label,
          permalink: tag.permalink,
        })
      })
    })

    return Array.from(counts.values())
      .sort((a, b) => b.count - a.count || a.label.localeCompare(b.label))
      .slice(0, 8)
  }, [items])

  const [featuredEntry, ...remainingEntries] = pageEntries

  return (
    <HtmlClassNameProvider
      className={`${ThemeClassNames.wrapper.blogPages} ${ThemeClassNames.page.blogListPage}`}
    >
      <BlogListPageMetadata {...props} />
      <BlogListPageStructuredData {...props} />
      <BlogLayout>
        <div className="site-blog-index">
          <div className="site-blog-index__header">
            <PageHeader
              description="Deep dives into model routing, inference engineering, performance breakthroughs, and the latest from the vLLM Semantic Router community."
              eyebrow="Blog"
              title="Engineering notes"
            />
          </div>

          <div className="site-blog-index__columns">
            <aside className="site-blog-index__rail">
              <section>
                <h2>Recent</h2>
                <ol className="site-blog-recent">
                  {items.slice(0, 6).map(({ content: BlogPostContent }) => (
                    <li key={BlogPostContent.metadata.permalink}>
                      <Link to={BlogPostContent.metadata.permalink}>
                        {BlogPostContent.metadata.title}
                      </Link>
                      <time dateTime={BlogPostContent.metadata.date}>
                        {formatDate(BlogPostContent.metadata.date)}
                      </time>
                    </li>
                  ))}
                </ol>
              </section>

              {tagSummaries.length > 0 && (
                <section className="site-blog-tags">
                  <h2>Tags</h2>
                  <ul>
                    {tagSummaries.map(tag => (
                      <li key={tag.permalink}>
                        <Link to={tag.permalink}>
                          <span aria-hidden="true">#</span>
                          {' '}
                          {tag.label}
                          {' '}
                          <small>{`(${tag.count})`}</small>
                        </Link>
                      </li>
                    ))}
                  </ul>
                </section>
              )}
            </aside>

            <section aria-live="polite" className="site-blog-index__feed">
              {featuredEntry
                ? (
                    <>
                      <BlogCard featured entry={featuredEntry} />
                      {remainingEntries.length > 0 && (
                        <div className="site-blog-card-grid">
                          {remainingEntries.map(entry => (
                            <BlogCard entry={entry} key={entry.permalink} />
                          ))}
                        </div>
                      )}
                    </>
                  )
                : (
                    <div className="site-blog-empty">
                      <h2>No posts found</h2>
                      <p>Check back soon for new engineering notes.</p>
                    </div>
                  )}
              <BlogListPaginator metadata={metadata} />
            </section>
          </div>
        </div>
      </BlogLayout>
    </HtmlClassNameProvider>
  )
}
