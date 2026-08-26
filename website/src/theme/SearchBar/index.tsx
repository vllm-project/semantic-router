import type { SearchResult } from '@easyops-cn/docusaurus-search-local/dist/client/client/theme/searchByWorker'
import type { ReactNode } from 'react'
import { useHistory, useLocation } from '@docusaurus/router'
import { translate } from '@docusaurus/Translate'
import useDocusaurusContext from '@docusaurus/useDocusaurusContext'
import useIsBrowser from '@docusaurus/useIsBrowser'
/*
 * Deep import into the search plugin's `dist/` tree.
 *
 * `@easyops-cn/docusaurus-search-local` ships no `exports` map in its
 * package.json, so every file under `dist/` stays a resolvable entry point for
 * both Node and webpack. This is a supported resolution rather than a hole
 * punched through an encapsulation boundary.
 *
 * It is also the only way in. The plugin's single public client surface is
 * `@theme/SearchBar` — precisely the component this file swizzles away — and
 * the worker handles live nowhere else. If a future release adds an `exports`
 * map this import is the first thing that breaks; the fix then is to vendor
 * these two functions, not to fork the whole search theme.
 *
 * The plugin emits no declarations, so `./searchByWorker.d.ts` next to this
 * file types the module.
 */
import {
  fetchIndexesByWorker,
  searchByWorker,
} from '@easyops-cn/docusaurus-search-local/dist/client/client/theme/searchByWorker'
import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { createPortal } from 'react-dom'

/**
 * Command-palette search.
 *
 * Replaces the search theme's anchored autocomplete dropdown with a ⌘K modal.
 * The index and the ranking are unchanged — this only swaps the surface the
 * results are presented on, so search stays offline-capable and needs no
 * hosted service.
 *
 * The local index is only built for production output: under `docusaurus
 * start` the worker short-circuits to an empty result set, so the palette says
 * so rather than looking broken.
 */

/*
 * Higher than the plugin's default of 8. Grouped results spend rows on section
 * headers, and a limit of 8 split across sections leaves each one looking
 * thinner than the index actually is.
 */
const RESULT_LIMIT = 12

/* Enough to outrun a fast typist without making the list feel laggy. */
const DEBOUNCE_MS = 110

/*
 * Content hits are indexed as whole paragraphs. Showing one from its first
 * character usually truncates before reaching the match, so the row shows a
 * window centred on the match instead — the same idea as the plugin's
 * `searchResultContextMaxLength`, at a width that suits a 36rem panel.
 */
const SNIPPET_WINDOW = 130

/*
 * The plugin's SearchDocumentType, and what each variant puts in a document
 * (see dist/server/server/utils/scanDocuments.js):
 *
 *   Title       t = page title,  b = breadcrumb
 *   Heading     t = heading,     p = page
 *   Description t = description, s = page title, p = page
 *   Keywords    t = keywords,    s = page title, p = page
 *   Content     t = paragraph,   s = heading,    p = page, h = hash
 */
const TYPE_TITLE = 0
const TYPE_KEYWORDS = 3
const TYPE_CONTENT = 4

type SectionId = 'docs' | 'blog' | 'pages'

interface Section {
  id: SectionId
  label: string
}

/*
 * The index covers the docs and the blog and nothing else: docusaurus.config.ts
 * sets `indexPages: false`, and the research and community routes are all
 * `src/pages/*` React pages. Research and Community therefore get no chip —
 * they can never produce a hit.
 *
 * `pages` is the honest landing spot for anything that is neither, which today
 * is nothing. It exists so that flipping `indexPages` on stays a config change:
 * the new routes get a neutral chip instead of being mislabelled "Docs".
 */
const SECTIONS: Record<SectionId, Section> = {
  docs: { id: 'docs', label: translate({ id: 'theme.SearchBar.section.docs', message: 'Docs' }) },
  blog: { id: 'blog', label: translate({ id: 'theme.SearchBar.section.blog', message: 'Blog' }) },
  pages: { id: 'pages', label: translate({ id: 'theme.SearchBar.section.pages', message: 'Pages' }) },
}

/** Page title, section and sub-section for one hit, as separate parts. */
interface Trail {
  /** Navbar section and sidebar categories above the page. */
  ancestors: string[]
  /** The page the hit lives on, when the hit is not the page itself. */
  page: string | null
  /** The heading the hit sits under, when it is not the page title. */
  section: string | null
}

interface Row {
  result: SearchResult
  /** The line the reader matched on. */
  title: string
  trail: Trail
  /** Position in render order, which is also arrow-key order. */
  index: number
}

interface Group {
  section: Section
  rows: Row[]
}

/**
 * A route reduced to its base- and locale-neutral form: `/docs/intro`, `/blog`,
 * `/`.
 *
 * Mirrors the normalisation in theme/Root.tsx, which is what decides the
 * `.site-root--home` class the landing-page rules in search.css key off. Both
 * have to agree, or the palette and its trigger disagree about which page is
 * the landing page.
 */
function normalizeRoute(url: string, baseUrl: string): string {
  const withoutBase = baseUrl !== '/' && url.startsWith(baseUrl)
    ? url.slice(baseUrl.length - 1)
    : url
  const path = withoutBase
    .replace(/\/+$/, '')
    .replace(/^\/zh-Hans(?=\/|$)/i, '')
  return path === '' ? '/' : path
}

/**
 * Which part of the site a hit belongs to, from its route.
 *
 * The two prefixes mirror `docsRouteBasePath` and `blogRouteBasePath` in
 * docusaurus.config.ts. The plugin does not surface them at runtime — its
 * generated constants carry only the index URL, the language and the limits —
 * so they are repeated here, and this is the place to change if either moves.
 */
function sectionFor(url: string, baseUrl: string): SectionId {
  const path = normalizeRoute(url, baseUrl)

  if (path === '/docs' || path.startsWith('/docs/')) {
    return 'docs'
  }
  if (path === '/blog' || path.startsWith('/blog/')) {
    return 'blog'
  }
  return 'pages'
}

function firstMatchAt(text: string, tokens: string[]): number {
  const lower = text.toLowerCase()
  let best = -1
  for (const token of tokens) {
    if (!token) {
      continue
    }
    const at = lower.indexOf(token.toLowerCase())
    if (at >= 0 && (best < 0 || at < best)) {
      best = at
    }
  }
  return best
}

function windowAroundMatch(text: string, tokens: string[]): string {
  if (text.length <= SNIPPET_WINDOW) {
    return text
  }
  const at = firstMatchAt(text, tokens)
  if (at < 0) {
    return `${text.slice(0, SNIPPET_WINDOW).trimEnd()}…`
  }
  // A short run-up keeps the match off the left edge, where it would read as
  // the start of a sentence rather than as a match.
  const start = Math.max(0, at - Math.round(SNIPPET_WINDOW / 4))
  const end = Math.min(text.length, start + SNIPPET_WINDOW)
  return [
    start > 0 ? '…' : '',
    text.slice(start, end).trim(),
    end < text.length ? '…' : '',
  ].join('')
}

/** The line the reader matched on, which differs by document type. */
function titleOf(result: SearchResult): string {
  const { document: hit, type } = result

  // Keywords documents hold a raw comma-separated list; the plugin shows the
  // page title for them instead, and so do we.
  if (type === TYPE_KEYWORDS) {
    return hit.s ?? hit.t
  }
  if (type === TYPE_CONTENT) {
    return windowAroundMatch(hit.t, result.tokens)
  }
  return hit.t
}

function trailFor(result: SearchResult, title: string): Trail {
  const { document: hit, page, type } = result

  // A page-title hit is already the page, so all it can add is what sits above
  // it in the sidebar.
  if (type === TYPE_TITLE || !page) {
    return { ancestors: (hit.b ?? []).filter(Boolean), page: null, section: null }
  }

  // `s` is the page title on description and keyword hits, and the enclosing
  // heading on content hits. Only the latter says anything the page line does
  // not already say.
  const section = hit.s && hit.s !== page.t && hit.s !== title ? hit.s : null

  return {
    ancestors: (page.b ?? []).filter(Boolean),
    page: page.t && page.t !== title ? page.t : null,
    section,
  }
}

/**
 * Results as the panel renders them: grouped by section, and flattened into
 * the row order the arrow keys walk.
 *
 * Groups appear in order of their best hit rather than in a fixed order, so
 * the plugin's ranking still decides which row sits under the cursor when the
 * palette opens.
 */
function buildRows(results: SearchResult[], baseUrl: string): { groups: Group[], rows: Row[] } {
  const groups: Group[] = []
  const byId = new Map<SectionId, Group>()

  for (const result of results) {
    const id = sectionFor(result.document.u, baseUrl)
    let group = byId.get(id)
    if (!group) {
      group = { section: SECTIONS[id], rows: [] }
      byId.set(id, group)
      groups.push(group)
    }
    const title = titleOf(result)
    group.rows.push({ result, title, trail: trailFor(result, title), index: -1 })
  }

  const rows = groups.flatMap(group => group.rows)
  rows.forEach((r, index) => {
    r.index = index
  })
  return { groups, rows }
}

function highlight(text: string, tokens: string[]): ReactNode {
  const needles = tokens.filter(Boolean).map(token => token.toLowerCase())
  if (needles.length === 0) {
    return text
  }

  const lower = text.toLowerCase()
  const spans: Array<[number, number]> = []
  for (const needle of needles) {
    let from = 0
    let at = lower.indexOf(needle, from)
    while (at >= 0) {
      spans.push([at, at + needle.length])
      from = at + needle.length
      at = lower.indexOf(needle, from)
    }
  }
  if (spans.length === 0) {
    return text
  }

  // Overlapping needles ("route", "router") would otherwise nest <mark>s.
  spans.sort((a, b) => a[0] - b[0])
  const merged: Array<[number, number]> = []
  for (const span of spans) {
    const last = merged[merged.length - 1]
    if (last && span[0] <= last[1]) {
      last[1] = Math.max(last[1], span[1])
    }
    else {
      merged.push([...span])
    }
  }

  const out: ReactNode[] = []
  let cursor = 0
  merged.forEach(([start, end], index) => {
    if (start > cursor) {
      out.push(text.slice(cursor, start))
    }
    out.push(<mark key={index}>{text.slice(start, end)}</mark>)
    cursor = end
  })
  if (cursor < text.length) {
    out.push(text.slice(cursor))
  }
  return out
}

/** True when the element is still in the document and actually rendered. */
function isVisible(element: HTMLElement | null): element is HTMLElement {
  return !!element && element.isConnected && element.getClientRects().length > 0
}

const STRINGS = {
  placeholder: translate({
    id: 'theme.SearchBar.placeholder',
    message: 'Search the docs and blog…',
    description: 'Placeholder in the command palette search field',
  }),
  results: translate({
    id: 'theme.SearchBar.resultsLabel',
    message: 'Search results',
    description: 'Accessible name of the command palette result list',
  }),
  idleTitle: translate({
    id: 'theme.SearchBar.idleTitle',
    message: 'Search the documentation and blog',
    description: 'Headline of the command palette before anything is typed',
  }),
  idleHint: translate({
    id: 'theme.SearchBar.idleHint',
    message: 'Results are grouped by section. ↵ opens the highlighted one.',
    description: 'Sub-line of the command palette before anything is typed',
  }),
  loading: translate({
    id: 'theme.SearchBar.loading',
    message: 'Loading the search index…',
    description: 'Shown while the search index is being fetched',
  }),
  unavailableTitle: translate({
    id: 'theme.SearchBar.unavailableTitle',
    message: 'Search is unavailable',
    description: 'Headline shown when the search index could not be fetched',
  }),
  unavailableHint: translate({
    id: 'theme.SearchBar.unavailableHint',
    message: 'The index could not be loaded. Check your connection and reload the page.',
    description: 'Sub-line shown when the search index could not be fetched',
  }),
  emptyHint: translate({
    id: 'theme.SearchBar.emptyHint',
    message: 'Try a shorter or more general term.',
    description: 'Advice shown when a query matches nothing',
  }),
  seeAll: translate({
    id: 'theme.SearchBar.seeAllShort',
    message: 'See all results',
    description: 'Row that leaves the palette for the standalone /search page',
  }),
  navigate: translate({ id: 'theme.SearchBar.hintNavigate', message: 'navigate' }),
  openHint: translate({ id: 'theme.SearchBar.hintOpen', message: 'open' }),
  closeHint: translate({ id: 'theme.SearchBar.hintClose', message: 'close' }),
}

type IndexState = 'idle' | 'loading' | 'ready' | 'failed'

export default function SearchBar(): ReactNode {
  const isBrowser = useIsBrowser()
  const history = useHistory()
  const location = useLocation()
  const { siteConfig: { baseUrl } } = useDocusaurusContext()

  /*
   * The landing page carries no search at all: no trigger (search.css hides
   * the navbar container), and no ⌘K or "/" either. A shortcut that works on a
   * page with no visible control is the inconsistency, not the fix — this
   * matches how the color-mode toggle is handled there, which is hidden with
   * no keyboard route back to it.
   *
   * Router state, not the DOM: this is resolved during SSR too, so there is no
   * hydration window where the shortcuts are briefly live here.
   */
  const isLandingPage = normalizeRoute(location.pathname || '/', baseUrl) === '/'

  const [open, setOpen] = useState(false)
  const [query, setQuery] = useState('')
  const [term, setTerm] = useState('')
  const [results, setResults] = useState<SearchResult[]>([])
  const [cursor, setCursor] = useState(0)
  const [indexState, setIndexState] = useState<IndexState>('idle')
  const [searching, setSearching] = useState(false)

  const inputRef = useRef<HTMLInputElement>(null)
  const listRef = useRef<HTMLDivElement>(null)
  const triggerRef = useRef<HTMLButtonElement>(null)
  const requestId = useRef(0)
  /*
   * Where focus was when the palette opened. The trigger is not always the
   * answer: ⌘K works from anywhere, and the trigger is hidden outright on the
   * landing page, where focusing it would silently drop focus to <body>.
   */
  const returnFocusTo = useRef<HTMLElement | null>(null)

  const isMac = useMemo(
    () => isBrowser && /mac|iphone|ipad/i.test(navigator.platform || navigator.userAgent),
    [isBrowser],
  )

  const { groups, rows } = useMemo(() => buildRows(results, baseUrl), [results, baseUrl])
  // The "see all results" row sits one past the last hit and is selectable, so
  // the keyboard can reach it even though Tab is trapped in the input.
  const seeAllIndex = rows.length
  const rowCount = rows.length > 0 ? rows.length + 1 : 0
  const debouncing = term !== query.trim()

  const close = useCallback(() => {
    setOpen(false)
    const target = returnFocusTo.current
    if (isVisible(target)) {
      target.focus()
    }
    else if (isVisible(triggerRef.current)) {
      triggerRef.current.focus()
    }
  }, [])

  const openPalette = useCallback(() => {
    // <body> means nothing had focus — a click in Safari, which does not focus
    // buttons. Fall back to the trigger in close() rather than parking focus
    // at the top of the document.
    const active = document.activeElement as HTMLElement | null
    returnFocusTo.current = active && active !== document.body ? active : null
    setOpen(true)
  }, [])

  /*
   * Load the index on first open rather than on mount — it is by far the
   * largest asset the search ships and most readers never open the palette.
   *
   * A failure is terminal for the page: the worker memoises the rejected fetch
   * promise, so retrying would resolve to the same rejection. The palette says
   * so instead of reporting an empty index as "no results".
   */
  useEffect(() => {
    if (!open || indexState !== 'idle') {
      return
    }
    let cancelled = false
    setIndexState('loading')
    fetchIndexesByWorker(baseUrl, '')
      .then(() => {
        if (!cancelled) {
          setIndexState('ready')
        }
      })
      .catch(() => {
        if (!cancelled) {
          setIndexState('failed')
        }
      })
    return () => {
      cancelled = true
    }
  }, [open, indexState, baseUrl])

  // Debounce the typed value into the term the worker actually searches for.
  useEffect(() => {
    const trimmed = query.trim()
    if (!trimmed) {
      setTerm('')
      return
    }
    const timer = window.setTimeout(() => setTerm(trimmed), DEBOUNCE_MS)
    return () => window.clearTimeout(timer)
  }, [query])

  useEffect(() => {
    if (!open) {
      return
    }
    if (!term) {
      // Invalidate any in-flight request so a late response cannot repopulate
      // a list the reader has just cleared.
      requestId.current += 1
      setResults([])
      setCursor(0)
      setSearching(false)
      return
    }

    const id = ++requestId.current
    setSearching(true)
    searchByWorker(baseUrl, '', term, RESULT_LIMIT)
      .then((found) => {
        if (id === requestId.current) {
          setResults(found ?? [])
          setCursor(0)
          setSearching(false)
        }
      })
      .catch(() => {
        if (id === requestId.current) {
          setResults([])
          setCursor(0)
          setSearching(false)
        }
      })
  }, [term, open, baseUrl])

  // Results can shrink under a cursor that was parked further down.
  useEffect(() => {
    setCursor(current => Math.min(current, Math.max(rowCount - 1, 0)))
  }, [rowCount])

  const go = useCallback((result: SearchResult) => {
    const { document: hit } = result
    close()
    history.push(hit.u + (hit.h ?? ''))
  }, [close, history])

  const goToSearchPage = useCallback(() => {
    if (!term) {
      return
    }
    close()
    history.push(`${baseUrl}search?q=${encodeURIComponent(term)}`)
  }, [baseUrl, close, history, term])

  // A client-side navigation onto the landing page while the palette is open —
  // browser back, say. Search is off there, so the panel goes with the page.
  useEffect(() => {
    if (isLandingPage && open) {
      close()
    }
  }, [isLandingPage, open, close])

  // Global shortcuts: ⌘K / Ctrl+K anywhere, and "/" outside a text field.
  useEffect(() => {
    if (!isBrowser || isLandingPage) {
      return
    }
    function onKeyDown(event: KeyboardEvent) {
      // Auto-repeat from a held key would otherwise flicker the palette.
      if (event.repeat) {
        return
      }
      const target = event.target as HTMLElement | null
      const typing = !!target && (
        target.isContentEditable
        || /^(?:INPUT|TEXTAREA|SELECT)$/.test(target.tagName)
      )

      if ((event.key === 'k' || event.key === 'K') && (event.metaKey || event.ctrlKey)) {
        event.preventDefault()
        if (open) {
          close()
        }
        else {
          openPalette()
        }
        return
      }
      // Bare "/" only: Ctrl+/ and friends belong to the browser and the editor.
      if (event.key === '/' && !typing && !event.metaKey && !event.ctrlKey && !event.altKey) {
        event.preventDefault()
        openPalette()
      }
    }
    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [isBrowser, isLandingPage, open, close, openPalette])

  /*
   * Hold the page still behind the modal. Removing the scrollbar would
   * otherwise widen the layout by its width and shift the whole page sideways
   * as the palette opens.
   */
  useEffect(() => {
    if (!isBrowser || !open) {
      return
    }
    const { body } = document
    const gutter = window.innerWidth - document.documentElement.clientWidth
    const previousOverflow = body.style.overflow
    const previousPadding = body.style.paddingRight
    body.style.overflow = 'hidden'
    if (gutter > 0) {
      body.style.paddingRight = `${gutter}px`
    }
    inputRef.current?.focus()
    return () => {
      body.style.overflow = previousOverflow
      body.style.paddingRight = previousPadding
    }
  }, [open, isBrowser])

  useEffect(() => {
    if (!open) {
      return
    }
    const selected = listRef.current?.querySelector('[aria-selected="true"]')
    selected?.scrollIntoView({ block: 'nearest' })
  }, [cursor, open])

  function onModalKeyDown(event: React.KeyboardEvent) {
    if (event.key === 'Escape') {
      event.preventDefault()
      close()
      return
    }
    if (event.key === 'ArrowDown') {
      event.preventDefault()
      setCursor(current => Math.min(current + 1, Math.max(rowCount - 1, 0)))
      return
    }
    if (event.key === 'ArrowUp') {
      event.preventDefault()
      setCursor(current => Math.max(current - 1, 0))
      return
    }
    if (event.key === 'Home' && rowCount > 0) {
      event.preventDefault()
      setCursor(0)
      return
    }
    if (event.key === 'End' && rowCount > 0) {
      event.preventDefault()
      setCursor(rowCount - 1)
      return
    }
    if (event.key === 'Enter') {
      event.preventDefault()
      const row = rows[cursor]
      if (row) {
        go(row.result)
      }
      else {
        goToSearchPage()
      }
      return
    }
    // Single-field dialog: keep Tab inside it rather than leaking to the page.
    if (event.key === 'Tab') {
      event.preventDefault()
      inputRef.current?.focus()
    }
  }

  function renderTrail(trail: Trail): ReactNode {
    if (trail.ancestors.length === 0 && !trail.page && !trail.section) {
      return null
    }
    return (
      <span className="site-search-hit__trail">
        {trail.ancestors.length > 0 && (
          <span className="site-search-hit__ancestors">{trail.ancestors.join(' › ')}</span>
        )}
        {trail.page && <span className="site-search-hit__page">{trail.page}</span>}
        {trail.section && <span className="site-search-hit__section">{trail.section}</span>}
      </span>
    )
  }

  function renderState(): ReactNode {
    if (indexState === 'failed') {
      return (
        <div className="site-search-state">
          <p className="site-search-state__title">{STRINGS.unavailableTitle}</p>
          <p className="site-search-state__hint">{STRINGS.unavailableHint}</p>
        </div>
      )
    }
    if (!query.trim()) {
      return (
        <div className="site-search-state">
          <p className="site-search-state__title">{STRINGS.idleTitle}</p>
          <p className="site-search-state__hint">{STRINGS.idleHint}</p>
        </div>
      )
    }
    if (indexState !== 'ready' || searching || debouncing) {
      return (
        <div className="site-search-state">
          <span aria-hidden="true" className="site-search-state__spinner" />
          <p className="site-search-state__hint">{STRINGS.loading}</p>
        </div>
      )
    }
    return (
      <div className="site-search-state">
        <p className="site-search-state__title">
          No results for “
          {term}
          ”
        </p>
        <p className="site-search-state__hint">{STRINGS.emptyHint}</p>
        <button className="site-search-state__action" onClick={goToSearchPage} type="button">
          {STRINGS.seeAll}
        </button>
      </div>
    )
  }

  /*
   * Render nothing on the landing page rather than leaning on the CSS alone:
   * no button in the accessibility tree, and no way for the palette to exist.
   * search.css still hides the container, which Docusaurus renders around this
   * component and which would otherwise leave a stray gap in the navbar.
   */
  if (isLandingPage) {
    return null
  }

  const modal = open && isBrowser
    ? createPortal(
        <div
          aria-label={STRINGS.placeholder}
          aria-modal="true"
          className="site-search-scrim"
          onMouseDown={(event) => {
            if (event.target === event.currentTarget) {
              close()
            }
          }}
          role="dialog"
        >
          {/* Key handling sits on the panel so it works wherever focus lands inside it. */}
          <div className="site-search-modal" onKeyDown={onModalKeyDown} role="presentation">
            <div className="site-search-modal__field">
              <svg
                aria-hidden="true"
                className="site-search-modal__icon"
                fill="none"
                height="17"
                stroke="currentColor"
                strokeLinecap="round"
                strokeWidth="2"
                viewBox="0 0 24 24"
                width="17"
              >
                <circle cx="11" cy="11" r="7" />
                <path d="M20 20l-3.5-3.5" />
              </svg>
              <input
                aria-activedescendant={rowCount > 0 ? `site-search-row-${cursor}` : undefined}
                aria-autocomplete="list"
                aria-controls="site-search-results"
                autoComplete="off"
                className="site-search-modal__input"
                onChange={event => setQuery(event.target.value)}
                placeholder={STRINGS.placeholder}
                ref={inputRef}
                spellCheck={false}
                type="text"
                value={query}
              />
            </div>

            <div
              aria-label={STRINGS.results}
              className="site-search-modal__results"
              id="site-search-results"
              ref={listRef}
              role="listbox"
            >
              {/* role="group" on the section itself, so every child of the
                  listbox is an option or a group, as ARIA requires. */}
              {groups.map(group => (
                <section
                  aria-labelledby={`site-search-group-${group.section.id}`}
                  className="site-search-group"
                  key={group.section.id}
                  role="group"
                >
                  <p className="site-search-group__head" id={`site-search-group-${group.section.id}`}>
                    <span className={`site-search-chip site-search-chip--${group.section.id}`}>
                      {group.section.label}
                    </span>
                    <span className="site-search-group__count">{group.rows.length}</span>
                  </p>
                  {group.rows.map(row => (
                    <div
                      aria-selected={row.index === cursor}
                      className="site-search-hit"
                      id={`site-search-row-${row.index}`}
                      key={`${row.result.document.i}-${row.result.type}`}
                      onClick={() => go(row.result)}
                      onMouseMove={() => setCursor(row.index)}
                      role="option"
                    >
                      <span aria-hidden="true" className="site-search-hit__kind">
                        {row.result.type === TYPE_TITLE ? '¶' : '#'}
                      </span>
                      <span className="site-search-hit__text">
                        <span className="site-search-hit__title">
                          {highlight(row.title, row.result.tokens)}
                        </span>
                        {renderTrail(row.trail)}
                      </span>
                      <span aria-hidden="true" className="site-search-hit__enter">↵</span>
                    </div>
                  ))}
                </section>
              ))}

              {rows.length > 0 && (
                <div
                  aria-selected={cursor === seeAllIndex}
                  className="site-search-seeall"
                  id={`site-search-row-${seeAllIndex}`}
                  onClick={goToSearchPage}
                  onMouseMove={() => setCursor(seeAllIndex)}
                  role="option"
                >
                  <span className="site-search-seeall__label">
                    {STRINGS.seeAll}
                    {' “'}
                    {term}
                    ”
                  </span>
                  <span aria-hidden="true" className="site-search-hit__enter">↵</span>
                </div>
              )}

              {rows.length === 0 && renderState()}
            </div>

            {/* The only place keys are named, and the only place counting
                happens is the group headers. Each fact appears once. */}
            <div className="site-search-modal__footer">
              <span>
                <kbd>↑</kbd>
                <kbd>↓</kbd>
                {' '}
                {STRINGS.navigate}
              </span>
              <span>
                <kbd>↵</kbd>
                {' '}
                {STRINGS.openHint}
              </span>
              <span>
                <kbd>esc</kbd>
                {' '}
                {STRINGS.closeHint}
              </span>
            </div>
          </div>
        </div>,
        document.body,
      )
    : null

  return (
    <>
      <button
        aria-haspopup="dialog"
        aria-label={STRINGS.placeholder}
        className="site-search-trigger"
        onClick={openPalette}
        ref={triggerRef}
        type="button"
      >
        <svg
          aria-hidden="true"
          fill="none"
          height="15"
          stroke="currentColor"
          strokeLinecap="round"
          strokeWidth="2"
          viewBox="0 0 24 24"
          width="15"
        >
          <circle cx="11" cy="11" r="7" />
          <path d="M20 20l-3.5-3.5" />
        </svg>
        <span className="site-search-trigger__label">Search</span>
        <kbd className="site-search-trigger__key">{isMac ? '⌘K' : 'Ctrl K'}</kbd>
      </button>
      {modal}
    </>
  )
}
