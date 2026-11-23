import React, { useCallback, useEffect, useRef, useState } from 'react'
import clsx from 'clsx'
import Translate, { translate } from '@docusaurus/Translate'
import Layout from '@theme/Layout'
import Head from '@docusaurus/Head'
import BrowserOnly from '@docusaurus/BrowserOnly'
import useDocusaurusContext from '@docusaurus/useDocusaurusContext'
import { PageIntro, PillLink } from '@site/src/components/site/Chrome'
import { SITE_SOCIAL_PREVIEW_IMAGE_PATH } from '@site/src/data/socialPreview'
import styles from './index.module.css'

const MOBILE_BREAKPOINT = 768
const MAX_SPREAD_WIDTH = 1400
const SPREAD_GAP = 16
const PDF_PAGE_RATIO = Math.SQRT2
const PAGINATION_HEIGHT = 56
const VIEWER_VERTICAL_PADDING = 40
const MIN_SPREAD_PAGE_WIDTH = 600
const SPREAD_FILL_THRESHOLD = 0.82
const MIN_SINGLE_PAGE_WIDTH = 320
const MAX_SINGLE_PAGE_WIDTH = 980

interface PaperViewerPageProps {
  heroDescription: string
  metaDescription: string
  pdfUrl: string
  socialTitle?: string
  title: string
}

export interface PaperChapter {
  title: string
  description?: string
  page: number
}

interface PaperViewerProps {
  pdfUrl: string
  /**
   * Chapter starts, in page order, listed beside the page. A reader with
   * chapters shows one page at a time.
   */
  chapters?: PaperChapter[]
}

function contentWidth(element: HTMLElement): number {
  const style = window.getComputedStyle(element)
  return element.clientWidth - parseFloat(style.paddingLeft) - parseFloat(style.paddingRight)
}

function isKeyboardInputTarget(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement))
    return false

  if (target.isContentEditable)
    return true

  return target instanceof HTMLInputElement
    || target instanceof HTMLTextAreaElement
    || target instanceof HTMLSelectElement
}

function PaperViewerContent({ pdfUrl, chapters }: PaperViewerProps): JSX.Element {
  // Lazily load react-pdf only in the browser to avoid SSG DOMMatrix errors.
  // eslint-disable-next-line @typescript-eslint/no-require-imports
  const { Document, Page, pdfjs } = require('react-pdf') as typeof import('react-pdf')
  // eslint-disable-next-line @typescript-eslint/no-require-imports
  require('react-pdf/dist/Page/AnnotationLayer.css')
  // eslint-disable-next-line @typescript-eslint/no-require-imports
  require('react-pdf/dist/Page/TextLayer.css')

  // Keep the scheme: on an http:// origin such as a local preview, a
  // protocol-relative URL loads http://unpkg.com, whose redirect to HTTPS has
  // no CORS header, so the module worker never loads.
  pdfjs.GlobalWorkerOptions.workerSrc = `https://unpkg.com/pdfjs-dist@${pdfjs.version}/build/pdf.worker.min.mjs`

  const [numPages, setNumPages] = useState<number>(0)
  const [pageNumber, setPageNumber] = useState<number>(1)
  const [pageWidth, setPageWidth] = useState<number>(600)
  const [isMobile, setIsMobile] = useState<boolean>(false)
  const [isSpread, setIsSpread] = useState<boolean>(true)
  const [loading, setLoading] = useState<boolean>(true)
  const [error, setError] = useState<boolean>(false)
  const viewerAreaRef = useRef<HTMLDivElement>(null)
  const chapterListRef = useRef<HTMLOListElement>(null)
  const hasChapters = Boolean(chapters?.length)

  useEffect(() => {
    const updateSize = () => {
      const vw = window.innerWidth
      const vh = window.innerHeight
      const mobile = vw <= MOBILE_BREAKPOINT
      setIsMobile(mobile)

      // The viewer's own box, not the window: the page around it decides how
      // much room it has, and pages sized from the window overflow it.
      const viewerWidth = viewerAreaRef.current ? contentWidth(viewerAreaRef.current) : vw

      if (mobile) {
        setIsSpread(false)
        setPageWidth(Math.max(1, Math.floor(viewerWidth)))
        return
      }

      const available = Math.min(viewerWidth, MAX_SPREAD_WIDTH)
      const spreadPageWidth = Math.floor(available / 2) - SPREAD_GAP
      const viewerHeight = Math.max(vh - PAGINATION_HEIGHT - VIEWER_VERTICAL_PADDING, 1)
      const spreadPageHeight = spreadPageWidth * PDF_PAGE_RATIO
      const spreadFillsViewport = spreadPageHeight >= viewerHeight * SPREAD_FILL_THRESHOLD
      const canUseSpread = !hasChapters
        && spreadPageWidth >= MIN_SPREAD_PAGE_WIDTH
        && spreadFillsViewport

      setIsSpread(canUseSpread)

      if (canUseSpread) {
        setPageWidth(spreadPageWidth)
        return
      }

      const widthByHeight = Math.floor(viewerHeight / PDF_PAGE_RATIO)
      setPageWidth(Math.max(
        MIN_SINGLE_PAGE_WIDTH,
        Math.min(viewerWidth, widthByHeight, MAX_SINGLE_PAGE_WIDTH),
      ))
    }

    updateSize()
    window.addEventListener('resize', updateSize)
    return () => window.removeEventListener('resize', updateSize)
  }, [hasChapters])

  useEffect(() => {
    setPageNumber((current) => {
      const maxPage = numPages > 0
        ? numPages
        : 1
      let next = Math.min(Math.max(current, 1), maxPage)
      if (isSpread && next % 2 === 0)
        next = Math.max(1, next - 1)
      return next
    })
  }, [isSpread, numPages])

  const onDocumentLoadSuccess = useCallback(({ numPages }: { numPages: number }) => {
    setNumPages(numPages)
    setLoading(false)
  }, [])

  const onDocumentLoadError = useCallback(() => {
    setError(true)
    setLoading(false)
  }, [])

  const step = isMobile || !isSpread
    ? 1
    : 2
  const scrollMobilePageIntoView = useCallback(() => {
    if (!isMobile)
      return

    requestAnimationFrame(() => viewerAreaRef.current?.scrollIntoView({ block: 'start' }))
  }, [isMobile])
  const goToPrev = useCallback(() => {
    setPageNumber(p => Math.max(1, p - step))
    scrollMobilePageIntoView()
  }, [scrollMobilePageIntoView, step])
  const goToNext = useCallback(() => {
    setPageNumber(p => Math.min(Math.max(1, numPages), p + step))
    scrollMobilePageIntoView()
  }, [numPages, scrollMobilePageIntoView, step])
  const goToPage = useCallback((page: number) => {
    setPageNumber(Math.min(Math.max(1, page), Math.max(1, numPages)))
    scrollMobilePageIntoView()
  }, [numPages, scrollMobilePageIntoView])

  const showChapters = hasChapters && !error
  const activeChapter = (chapters ?? []).reduce(
    (active, chapter, index) => (chapter.page <= pageNumber ? index : active),
    -1,
  )

  // Stacked above the page, the chapters scroll sideways: keep the current one
  // in view without scrolling the document.
  useEffect(() => {
    const list = chapterListRef.current
    const item = activeChapter >= 0 ? list?.children[activeChapter] : undefined
    if (!list || !(item instanceof HTMLElement) || list.scrollWidth <= list.clientWidth)
      return

    list.scrollTo({
      left: item.offsetLeft - (list.clientWidth - item.offsetWidth) / 2,
      behavior: 'smooth',
    })
  }, [activeChapter])

  const rightPage = pageNumber + 1
  const hasRight = isSpread && rightPage <= numPages
  const isNextDisabled = pageNumber >= numPages - (step - 1)
  const isDesktopSinglePage = !isMobile && !isSpread
  const documentClassName = isDesktopSinglePage
    ? `${styles.document} ${styles.documentSinglePage}`
    : styles.document

  useEffect(() => {
    if (loading || error)
      return undefined

    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.metaKey || event.ctrlKey || event.altKey || isKeyboardInputTarget(event.target))
        return

      if (event.key === 'ArrowLeft' && pageNumber > 1) {
        event.preventDefault()
        goToPrev()
      }

      if (event.key === 'ArrowRight' && !isNextDisabled) {
        event.preventDefault()
        goToNext()
      }
    }

    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [error, goToNext, goToPrev, isNextDisabled, loading, pageNumber])

  return (
    <div
      className={clsx(styles.viewerShell, showChapters && styles.viewerShellChapters)}
      tabIndex={-1}
      aria-keyshortcuts="ArrowLeft ArrowRight"
    >
      {showChapters && (
        <nav
          className={styles.chapters}
          aria-label={translate({
            id: 'paperViewer.chapters.label',
            message: 'Chapters',
          })}
        >
          <ol ref={chapterListRef}>
            {chapters?.map((chapter, index) => (
              <li key={chapter.page}>
                <button
                  type="button"
                  className={styles.chapter}
                  onClick={() => goToPage(chapter.page)}
                  disabled={loading}
                  aria-current={index === activeChapter ? 'true' : undefined}
                >
                  <span className={styles.chapterNumber}>{String(index + 1).padStart(2, '0')}</span>
                  <span className={styles.chapterTitle}>{chapter.title}</span>
                  {chapter.description && (
                    <span className={styles.chapterDescription}>{chapter.description}</span>
                  )}
                </button>
              </li>
            ))}
          </ol>
        </nav>
      )}

      <div ref={viewerAreaRef} className={styles.viewerArea}>
        {error
          ? (
              <div className={styles.fallback}>
                <p>
                  <Translate id="paperViewer.error.loadPreview">
                    Unable to load PDF preview.
                  </Translate>
                </p>
                <a href={pdfUrl} target="_blank" rel="noopener noreferrer">
                  <Translate id="paperViewer.error.openNewTab">
                    Click here to open the PDF in a new tab
                  </Translate>
                </a>
              </div>
            )
          : (
              <Document
                file={pdfUrl}
                onLoadSuccess={onDocumentLoadSuccess}
                onLoadError={onDocumentLoadError}
                loading={(
                  <div className={styles.loadingText}>
                    <Translate id="paperViewer.loading">Loading PDF…</Translate>
                  </div>
                )}
                className={documentClassName}
              >
                {isSpread
                  ? (
                      <div className={styles.pagesRow}>
                        <div className={styles.pageWrapper}>
                          <Page
                            pageNumber={pageNumber}
                            width={pageWidth}
                            renderTextLayer={true}
                            renderAnnotationLayer={true}
                          />
                        </div>
                        {hasRight && (
                          <div className={styles.pageWrapper}>
                            <Page
                              pageNumber={rightPage}
                              width={pageWidth}
                              renderTextLayer={true}
                              renderAnnotationLayer={true}
                            />
                          </div>
                        )}
                      </div>
                    )
                  : (
                      <div className={styles.pageWrapper}>
                        <Page
                          pageNumber={pageNumber}
                          width={pageWidth}
                          renderTextLayer={true}
                          renderAnnotationLayer={true}
                        />
                      </div>
                    )}
              </Document>
            )}
      </div>

      {!error && !loading && (
        <div className={styles.pagination}>
          <div />
          <div className={styles.paginationCenter}>
            <button
              className={styles.pageBtn}
              onClick={goToPrev}
              disabled={pageNumber <= 1}
              aria-label={translate({
                id: 'paperViewer.pagination.prevAria',
                message: 'Previous page',
              })}
            >
              ←
              {' '}
              <Translate id="paperViewer.pagination.prev">Prev</Translate>
            </button>
            <span className={styles.pageInfo}>
              {pageNumber}
              {hasRight ? `–${rightPage}` : ''}
              {' '}
              /
              {numPages}
            </span>
            <button
              className={styles.pageBtn}
              onClick={goToNext}
              disabled={isNextDisabled}
              aria-label={translate({
                id: 'paperViewer.pagination.nextAria',
                message: 'Next page',
              })}
            >
              <Translate id="paperViewer.pagination.next">Next</Translate>
              {' '}
              →
            </button>
          </div>
          <div />
        </div>
      )}
    </div>
  )
}

/** The reader without the page around it, for pages that frame it themselves. */
export function PaperViewer(props: PaperViewerProps): JSX.Element {
  return (
    <BrowserOnly
      fallback={(
        <div className={styles.loadingText}>
          <Translate id="paperViewer.loading">Loading PDF…</Translate>
        </div>
      )}
    >
      {() => <PaperViewerContent {...props} />}
    </BrowserOnly>
  )
}

export default function PaperViewerPage({
  heroDescription,
  metaDescription,
  pdfUrl,
  socialTitle,
  title,
}: PaperViewerPageProps): JSX.Element {
  const { siteConfig } = useDocusaurusContext()
  const ogImage = new URL(SITE_SOCIAL_PREVIEW_IMAGE_PATH, siteConfig.url).toString()
  const resolvedSocialTitle = socialTitle ?? `${title} — vLLM Semantic Router`

  return (
    <Layout
      title={title}
      description={metaDescription}
    >
      <Head>
        <meta property="og:title" content={resolvedSocialTitle} />
        <meta property="og:description" content={metaDescription} />
        <meta property="og:image" content={ogImage} />
        <meta property="og:type" content="article" />
        <meta name="twitter:card" content="summary_large_image" />
        <meta name="twitter:title" content={resolvedSocialTitle} />
        <meta name="twitter:description" content={metaDescription} />
        <meta name="twitter:image" content={ogImage} />
      </Head>
      <main className={styles.page}>
        <div className="site-shell-container">
          <div className={styles.hero}>
            <PageIntro
              label={(
                <Translate id="paperViewer.hero.label">Research document</Translate>
              )}
              title={title}
              description={heroDescription}
              actions={(
                <>
                  <PillLink href={pdfUrl} target="_blank" rel="noreferrer">
                    <Translate id="paperViewer.hero.download">Download PDF</Translate>
                  </PillLink>
                  <PillLink to="/publications" muted>
                    <Translate id="paperViewer.hero.backToPublications">
                      Research routes
                    </Translate>
                  </PillLink>
                </>
              )}
            />
          </div>
        </div>

        <div className="site-shell-container">
          <PaperViewer pdfUrl={pdfUrl} />
        </div>
      </main>
    </Layout>
  )
}
