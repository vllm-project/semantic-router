import React, { useMemo, useState } from 'react'
import Link from '@docusaurus/Link'
import useDocusaurusContext from '@docusaurus/useDocusaurusContext'
import useGlobalData from '@docusaurus/useGlobalData'
import {
  HtmlClassNameProvider,
  PageMetadata,
  ThemeClassNames,
} from '@docusaurus/theme-common'
import SearchMetadata from '@theme/SearchMetadata'
import BlogLayout from '@theme/BlogLayout'
import BlogListPaginator from '@theme/BlogListPaginator'
import BlogListPageStructuredData from '@theme/BlogListPage/StructuredData'
import type { Props } from '@theme/BlogListPage'
import {
  BLOG_SEARCH_INDEX_PLUGIN_NAME,
  type BlogSearchIndex,
  type BlogSearchIndexEntry,
} from '../../plugins/blogSearchIndex'

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
 * Normalizes a post from `props.items` into the same plain shape the search index
 * uses, so cards render identically whether they came from the current page or from
 * the site-wide index.
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

/**
 * Reads the site-wide post index published by the `blog-search-index` plugin.
 *
 * Returns `null` when the index is unavailable so callers can fall back to the posts
 * Docusaurus passed in props. Global data is read defensively rather than through
 * `usePluginData` because a missing entry must degrade, never throw.
 */
function useBlogSearchIndex(): BlogSearchIndexEntry[] | null {
  const globalData = useGlobalData()

  return useMemo(() => {
    const pluginData = globalData?.[BLOG_SEARCH_INDEX_PLUGIN_NAME]?.default as
      | BlogSearchIndex
      | undefined
    const posts = pluginData?.posts

    return Array.isArray(posts) && posts.length > 0 ? posts : null
  }, [globalData])
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
  const { date, description, image, permalink, readingTime, title } = entry
  const readTime = readingTimeLabel(readingTime)

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
        {featured && <span className="site-blog-card__eyebrow">Featured</span>}
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
  const [query, setQuery] = useState('')
  const normalizedQuery = query.trim().toLowerCase()
  const isSearching = normalizedQuery.length > 0
  const searchIndex = useBlogSearchIndex()

  // Posts for the page Docusaurus rendered. Used verbatim when not searching, and as
  // the fallback corpus if the site-wide index is ever unavailable.
  const pageEntries = useMemo(() => items.map(toEntry), [items])

  const searchableEntries = searchIndex ?? pageEntries

  const filteredEntries = useMemo(() => {
    if (!isSearching) {
      return pageEntries
    }

    return searchableEntries.filter((entry) => {
      const haystack = [
        entry.title,
        entry.description,
        entry.tags.join(' '),
      ].join(' ').toLowerCase()

      return haystack.includes(normalizedQuery)
    })
  }, [isSearching, normalizedQuery, pageEntries, searchableEntries])

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

  const [featuredEntry, ...remainingEntries] = filteredEntries

  return (
    <HtmlClassNameProvider
      className={`${ThemeClassNames.wrapper.blogPages} ${ThemeClassNames.page.blogListPage}`}
    >
      <BlogListPageMetadata {...props} />
      <BlogListPageStructuredData {...props} />
      <BlogLayout>
        <div className="site-blog-index">
          <header className="site-blog-index__header">
            <h1>Blog</h1>
            <label className="site-blog-search">
              <span aria-hidden="true" className="site-blog-search__icon">⌕</span>
              <input
                aria-label="Search blog posts"
                onChange={event => setQuery(event.target.value)}
                placeholder="Search blog posts..."
                type="search"
                value={query}
              />
            </label>
            <p>
              Deep dives into model routing, inference engineering, performance
              breakthroughs, and the latest from the vLLM Semantic Router
              community.
            </p>
          </header>

          <div className="site-blog-index__columns">
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
                      <p>Try a different title, topic, or tag.</p>
                    </div>
                  )}
              {!isSearching && <BlogListPaginator metadata={metadata} />}
            </section>

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
          </div>
        </div>
      </BlogLayout>
    </HtmlClassNameProvider>
  )
}
