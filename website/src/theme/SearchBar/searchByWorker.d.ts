/*
 * Types for the one search-plugin module this theme component reaches into.
 *
 * `@easyops-cn/docusaurus-search-local` compiles its client to plain `.js`
 * with no emitted declarations, so the deep import in `index.tsx` would
 * otherwise resolve to an implicit `any`. The shapes below are transcribed
 * from `dist/client/client/theme/worker.js` (the search result it posts back)
 * and `dist/client/shared/interfaces.js` (`SearchDocumentType`).
 *
 * Only the fields this palette reads are declared. Widen it here rather than
 * casting at the call site if more of the document is ever needed.
 */
declare module '@easyops-cn/docusaurus-search-local/dist/client/client/theme/searchByWorker' {
  /** One indexed unit: a page title, a heading, or a chunk of content. */
  export interface SearchDocument {
    /** Document id, unique within the index. */
    i: number
    /** Title, or the heading text for a heading hit. */
    t: string
    /** Route, without the fragment. */
    u: string
    /** Fragment for a heading hit, including the leading `#`. */
    h?: string
    /** Section: the heading this document sits under. */
    s?: string
    /** Breadcrumb trail above the page. */
    b?: string[]
    /** Id of the page document this one belongs to. */
    p?: number
  }

  export interface SearchResult {
    document: SearchDocument
    /** The owning page, present for everything but a Title hit. */
    page?: SearchDocument
    /** A `SearchDocumentType`: 0 Title, 1 Heading, 2 Description, 3 Keywords, 4 Content. */
    type: number
    /** The query as the plugin tokenized it, used for highlighting. */
    tokens: string[]
  }

  /** No-ops outside a production build, where no index has been emitted. */
  export function fetchIndexesByWorker(
    baseUrl: string,
    searchContext: string,
  ): Promise<void>

  /** Resolves to `[]` outside a production build. */
  export function searchByWorker(
    baseUrl: string,
    searchContext: string,
    input: string,
    limit: number,
  ): Promise<SearchResult[]>
}
