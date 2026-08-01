import type { Plugin } from '@docusaurus/types'

/**
 * Docusaurus hands `BlogListPage` only the posts for the page being rendered, so a
 * client-side filter over those props can never see the rest of the archive. This
 * plugin publishes a small, plain-data index of every published post as global data
 * so the blog list page can search the whole blog instead of the current page.
 *
 * The index is intentionally minimal: it carries only the fields the blog cards and
 * the search predicate need, because global data ships to the browser on every page.
 */

export const BLOG_SEARCH_INDEX_PLUGIN_NAME = 'blog-search-index'

const BLOG_CONTENT_PLUGIN_NAME = 'docusaurus-plugin-content-blog'
const BLOG_CONTENT_PLUGIN_ID = 'default'

export interface BlogSearchIndexEntry {
  permalink: string
  title: string
  /** ISO timestamp; global data is serialized to JSON, so `Date` cannot survive. */
  date: string
  description: string
  image?: string
  readingTime?: number
  tags: string[]
}

export interface BlogSearchIndex {
  posts: BlogSearchIndexEntry[]
}

/**
 * Shape of the parts of `blogPosts` this plugin reads. Declared locally and matched
 * defensively so an upstream change degrades to an empty index rather than failing
 * the production build — `BlogListPage` falls back to its own props in that case.
 */
interface LoadedBlogPost {
  metadata?: {
    permalink?: unknown
    title?: unknown
    date?: unknown
    description?: unknown
    readingTime?: unknown
    unlisted?: unknown
    tags?: unknown
    frontMatter?: { image?: unknown }
  }
}

function asString(value: unknown): string | undefined {
  return typeof value === 'string' && value.length > 0 ? value : undefined
}

function toISODate(value: unknown): string | undefined {
  if (value instanceof Date) {
    return Number.isNaN(value.getTime()) ? undefined : value.toISOString()
  }
  if (typeof value === 'string') {
    const parsed = new Date(value)
    return Number.isNaN(parsed.getTime()) ? undefined : parsed.toISOString()
  }
  return undefined
}

function toTagLabels(value: unknown): string[] {
  if (!Array.isArray(value)) {
    return []
  }

  return value
    .map((tag) => {
      if (typeof tag === 'string') {
        return tag
      }
      if (tag && typeof tag === 'object' && 'label' in tag) {
        return asString((tag as { label?: unknown }).label)
      }
      return undefined
    })
    .filter((label): label is string => typeof label === 'string')
}

function toIndexEntry(post: LoadedBlogPost): BlogSearchIndexEntry | undefined {
  const metadata = post?.metadata
  if (!metadata || metadata.unlisted === true) {
    return undefined
  }

  const permalink = asString(metadata.permalink)
  const title = asString(metadata.title)
  const date = toISODate(metadata.date)

  // A card is only useful if it can be linked, labelled and dated.
  if (!permalink || !title || !date) {
    return undefined
  }

  return {
    permalink,
    title,
    date,
    description: asString(metadata.description) ?? '',
    image: asString(metadata.frontMatter?.image),
    readingTime: typeof metadata.readingTime === 'number' ? metadata.readingTime : undefined,
    tags: toTagLabels(metadata.tags),
  }
}

export function buildBlogSearchIndex(blogPosts: unknown): BlogSearchIndex {
  if (!Array.isArray(blogPosts)) {
    return { posts: [] }
  }

  const posts = blogPosts
    .map(post => toIndexEntry(post as LoadedBlogPost))
    .filter((entry): entry is BlogSearchIndexEntry => entry !== undefined)
    // Newest first, matching the order Docusaurus paginates the blog in.
    .sort((a, b) => b.date.localeCompare(a.date))

  return { posts }
}

export default function blogSearchIndexPlugin(): Plugin {
  return {
    name: BLOG_SEARCH_INDEX_PLUGIN_NAME,

    // `allContentLoaded` is the only lifecycle that sees other plugins' content, and
    // it runs after every plugin has loaded, so the blog posts are guaranteed present.
    allContentLoaded({ allContent, actions }) {
      const blogContent = allContent?.[BLOG_CONTENT_PLUGIN_NAME]?.[BLOG_CONTENT_PLUGIN_ID] as
        | { blogPosts?: unknown }
        | undefined

      actions.setGlobalData(buildBlogSearchIndex(blogContent?.blogPosts))
    },
  }
}
