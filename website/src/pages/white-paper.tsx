import React, { useState, useCallback, useEffect } from 'react'
import Layout from '@theme/Layout'
import Head from '@docusaurus/Head'
import BrowserOnly from '@docusaurus/BrowserOnly'
import styles from './white-paper.module.css'

const PDF_URL = '/white-paper.pdf'

// Inner component: only rendered in the browser, avoids SSG DOMMatrix errors
function WhitePaperContent(): JSX.Element {
  // Lazily load react-pdf only in the browser to avoid SSG DOMMatrix errors
  // eslint-disable-next-line @typescript-eslint/no-require-imports
  const { Document, Page, pdfjs } = require('react-pdf') as typeof import('react-pdf')
  // eslint-disable-next-line @typescript-eslint/no-require-imports
  require('react-pdf/dist/Page/AnnotationLayer.css')
  // eslint-disable-next-line @typescript-eslint/no-require-imports
  require('react-pdf/dist/Page/TextLayer.css')

  // Configure PDF.js worker
  pdfjs.GlobalWorkerOptions.workerSrc = `//unpkg.com/pdfjs-dist@${pdfjs.version}/build/pdf.worker.min.mjs`

  const [numPages, setNumPages] = useState<number>(0)
  const [pageNumber, setPageNumber] = useState<number>(1)
  const [pageWidth, setPageWidth] = useState<number>(600)
  // Mobile: render by height so the page fills the screen exactly (no gray gap below)
  const [pageHeight, setPageHeight] = useState<number | undefined>(undefined)
  const [isMobile, setIsMobile] = useState<boolean>(false)
  const [loading, setLoading] = useState<boolean>(true)
  const [error, setError] = useState<boolean>(false)
  // Bottom bar height (px) used when computing available height on mobile
  const BOTTOM_BAR_H = 52

  // Hide the built-in Docusaurus navbar and footer for a full-screen viewer
  useEffect(() => {
    const navbar = document.querySelector('.navbar') as HTMLElement | null
    const footer = document.querySelector('footer') as HTMLElement | null
    if (navbar) navbar.style.display = 'none'
    if (footer) footer.style.display = 'none'
    return () => {
      if (navbar) navbar.style.display = ''
      if (footer) footer.style.display = ''
    }
  }, [])

  // Dynamically calculate page dimensions based on viewport
  useEffect(() => {
    const updateSize = () => {
      const vw = window.innerWidth
      const mobile = vw <= 768
      setIsMobile(mobile)
      if (mobile) {
        // Mobile: height-driven rendering — fill screen minus the bottom bar,
        // so there is zero gray gap between the PDF page and the nav bar.
        setPageHeight(window.innerHeight - BOTTOM_BAR_H)
        setPageWidth(0) // unused on mobile
      }
      else {
        // Desktop: two-page spread, each half of available width
        setPageHeight(undefined)
        const available = Math.min(vw - 120, 1400)
        setPageWidth(Math.floor(available / 2) - 16)
      }
    }
    updateSize()
    window.addEventListener('resize', updateSize)
    return () => window.removeEventListener('resize', updateSize)
  }, [BOTTOM_BAR_H])

  const onDocumentLoadSuccess = useCallback(({ numPages }: { numPages: number }) => {
    setNumPages(numPages)
    setLoading(false)
  }, [])

  const onDocumentLoadError = useCallback(() => {
    setError(true)
    setLoading(false)
  }, [])

  // Pagination: advance 1 page on mobile, 2 pages on desktop
  const step = isMobile
    ? 1
    : 2
  const goToPrev = () => setPageNumber(p => Math.max(1, p - step))
  const goToNext = () => setPageNumber(p => Math.min(numPages, p + step))

  // Right-page number (desktop two-page mode only)
  const rightPage = pageNumber + 1
  const hasRight = !isMobile && rightPage <= numPages

  return (
    <div className={styles.page}>
      {/* PDF viewer area */}
      <div className={styles.viewerArea}>
        {error
          ? (
              <div className={styles.fallback}>
                <p>Unable to load PDF preview.</p>
                <a href={PDF_URL} target="_blank" rel="noopener noreferrer">
                  Click here to open the PDF in a new tab
                </a>
              </div>
            )
          : (
              <Document
                file={PDF_URL}
                onLoadSuccess={onDocumentLoadSuccess}
                onLoadError={onDocumentLoadError}
                loading={<div className={styles.loadingText}>Loading PDF…</div>}
                className={styles.document}
              >
                {/* Two-page spread */}
                <div className={styles.pagesRow}>
                  <div className={styles.pageWrapper}>
                    <Page
                      pageNumber={pageNumber}
                      width={isMobile ? undefined : pageWidth}
                      height={isMobile ? pageHeight : undefined}
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
              </Document>
            )}
      </div>

      {/* Bottom control bar: three-column layout */}
      {!error && !loading && (
        <div className={styles.pagination}>
          {/* Left spacer */}
          <div />
          {/* Center: page navigation */}
          <div className={styles.paginationCenter}>
            <button
              className={styles.pageBtn}
              onClick={goToPrev}
              disabled={pageNumber <= 1}
              aria-label="Previous page"
            >
              ← Prev
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
              disabled={pageNumber + 1 >= numPages}
              aria-label="Next page"
            >
              Next →
            </button>
          </div>
          {/* Right spacer */}
          <div />
        </div>
      )}
    </div>
  )
}

export default function WhitePaper(): JSX.Element {
  const ogImage = 'https://vllm-semantic-router.com/img/vllm.png'
  return (
    <Layout
      title="White Paper"
      description="Signal Driven Decision Routing for Mixture-of-Modality Models — Official White Paper"
    >
      <Head>
        <meta property="og:title" content="White Paper — vLLM Semantic Router" />
        <meta property="og:description" content="Signal Driven Decision Routing for Mixture-of-Modality Models — Official White Paper" />
        <meta property="og:image" content={ogImage} />
        <meta property="og:type" content="article" />
        <meta name="twitter:card" content="summary" />
        <meta name="twitter:title" content="White Paper — vLLM Semantic Router" />
        <meta name="twitter:description" content="Signal Driven Decision Routing for Mixture-of-Modality Models — Official White Paper" />
        <meta name="twitter:image" content={ogImage} />
      </Head>
      {/* BrowserOnly prevents SSG from executing browser-only APIs (e.g. DOMMatrix) */}
      <BrowserOnly fallback={<div className={styles.loadingText}>Loading PDF…</div>}>
        {() => <WhitePaperContent />}
      </BrowserOnly>
    </Layout>
  )
}
