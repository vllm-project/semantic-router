import React, { useState, useCallback, useEffect } from 'react'
import Translate, { translate } from '@docusaurus/Translate'
import Layout from '@theme/Layout'
import Head from '@docusaurus/Head'
import BrowserOnly from '@docusaurus/BrowserOnly'
import useDocusaurusContext from '@docusaurus/useDocusaurusContext'
import { PageIntro, PillLink } from '@site/src/components/site/Chrome'
import { SITE_SOCIAL_PREVIEW_IMAGE_PATH } from '@site/src/data/socialPreview'
import styles from './index.module.css'

const MOBILE_BREAKPOINT = 768
const MAX_SPREAD_VIEWPORT_WIDTH = 1400
const VIEWPORT_SIDE_PADDING = 120
const SPREAD_GAP = 16
const PDF_PAGE_RATIO = Math.SQRT2
const PAGINATION_HEIGHT = 56
const VIEWER_VERTICAL_PADDING = 40
const MIN_SPREAD_PAGE_WIDTH = 620
const SPREAD_FILL_THRESHOLD = 0.82
const MAX_SINGLE_PAGE_WIDTH = 980

interface PaperViewerPageProps {
  heroDescription: string
  metaDescription: string
  pdfUrl: string
  socialTitle?: string
  title: string
}

interface PaperViewerContentProps {
  pdfUrl: string
}

function PaperViewerContent({ pdfUrl }: PaperViewerContentProps): JSX.Element {
  // Lazily load react-pdf only in the browser to avoid SSG DOMMatrix errors.
  // eslint-disable-next-line @typescript-eslint/no-require-imports
  const { Document, Page, pdfjs } = require('react-pdf') as typeof import('react-pdf')
  // eslint-disable-next-line @typescript-eslint/no-require-imports
  require('react-pdf/dist/Page/AnnotationLayer.css')
  // eslint-disable-next-line @typescript-eslint/no-require-imports
  require('react-pdf/dist/Page/TextLayer.css')

  pdfjs.GlobalWorkerOptions.workerSrc = `//unpkg.com/pdfjs-dist@${pdfjs.version}/build/pdf.worker.min.mjs`

  const [numPages, setNumPages] = useState<number>(0)
  const [pageNumber, setPageNumber] = useState<number>(1)
  const [pageWidth, setPageWidth] = useState<number>(600)
  const [isMobile, setIsMobile] = useState<boolean>(false)
  const [isSpread, setIsSpread] = useState<boolean>(true)
  const [loading, setLoading] = useState<boolean>(true)
  const [error, setError] = useState<boolean>(false)

  useEffect(() => {
    const updateSize = () => {
      const vw = window.innerWidth
      const vh = window.innerHeight
      const mobile = vw <= MOBILE_BREAKPOINT
      setIsMobile(mobile)

      if (mobile) {
        setIsSpread(false)
        setPageWidth(Math.max(320, vw - 2))
        return
      }

      const available = Math.min(vw - VIEWPORT_SIDE_PADDING, MAX_SPREAD_VIEWPORT_WIDTH)
      const spreadPageWidth = Math.floor(available / 2) - SPREAD_GAP
      const viewerHeight = Math.max(vh - PAGINATION_HEIGHT - VIEWER_VERTICAL_PADDING, 1)
      const spreadPageHeight = spreadPageWidth * PDF_PAGE_RATIO
      const spreadFillsViewport = spreadPageHeight >= viewerHeight * SPREAD_FILL_THRESHOLD
      const canUseSpread = spreadPageWidth >= MIN_SPREAD_PAGE_WIDTH && spreadFillsViewport

      setIsSpread(canUseSpread)

      if (canUseSpread) {
        setPageWidth(spreadPageWidth)
        return
      }

      const widthByViewport = Math.min(vw - 96, MAX_SINGLE_PAGE_WIDTH)
      const widthByHeight = Math.floor(viewerHeight / PDF_PAGE_RATIO)
      setPageWidth(Math.max(320, Math.min(widthByViewport, widthByHeight)))
    }

    updateSize()
    window.addEventListener('resize', updateSize)
    return () => window.removeEventListener('resize', updateSize)
  }, [])

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
  const goToPrev = () => setPageNumber(p => Math.max(1, p - step))
  const goToNext = () => setPageNumber(p => Math.min(numPages, p + step))

  const rightPage = pageNumber + 1
  const hasRight = isSpread && rightPage <= numPages
  const isNextDisabled = pageNumber >= numPages - (step - 1)
  const isDesktopSinglePage = !isMobile && !isSpread
  const documentClassName = isDesktopSinglePage
    ? `${styles.document} ${styles.documentSinglePage}`
    : styles.document

  return (
    <div className={styles.viewerShell}>
      <div className={styles.viewerArea}>
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
                {isMobile
                  ? (
                      <div className={styles.mobileStack}>
                        {Array.from({ length: numPages }, (_, i) => (
                          <div key={i + 1} className={styles.pageWrapper}>
                            <Page
                              pageNumber={i + 1}
                              width={pageWidth}
                              renderTextLayer={true}
                              renderAnnotationLayer={true}
                            />
                          </div>
                        ))}
                      </div>
                    )
                  : isSpread
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

      {!error && !loading && !isMobile && (
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
          <BrowserOnly
            fallback={(
              <div className={styles.loadingText}>
                <Translate id="paperViewer.loading">Loading PDF…</Translate>
              </div>
            )}
          >
            {() => <PaperViewerContent pdfUrl={pdfUrl} />}
          </BrowserOnly>
        </div>
      </main>
    </Layout>
  )
}
