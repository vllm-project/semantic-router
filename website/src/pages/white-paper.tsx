import React, { useState, useCallback, useEffect } from 'react'
import Layout from '@theme/Layout'
import { Document, Page, pdfjs } from 'react-pdf'
import 'react-pdf/dist/Page/AnnotationLayer.css'
import 'react-pdf/dist/Page/TextLayer.css'
import styles from './white-paper.module.css'

// Configure PDF.js worker
pdfjs.GlobalWorkerOptions.workerSrc = `//unpkg.com/pdfjs-dist@${pdfjs.version}/build/pdf.worker.min.mjs`

const PDF_URL = '/vllm-semantic-router.pdf'

export default function WhitePaper(): JSX.Element {
  const [numPages, setNumPages] = useState<number>(0)
  const [pageNumber, setPageNumber] = useState<number>(1)
  const [pageWidth, setPageWidth] = useState<number>(600)
  const [isMobile, setIsMobile] = useState<boolean>(false)
  const [loading, setLoading] = useState<boolean>(true)
  const [error, setError] = useState<boolean>(false)

  // Hide the built-in Docusaurus footer
  useEffect(() => {
    const footer = document.querySelector('footer')
    if (footer) {
      footer.style.display = 'none'
    }
    return () => {
      if (footer) {
        footer.style.display = ''
      }
    }
  }, [])

  // Dynamically calculate page width based on viewport; single-page on mobile
  useEffect(() => {
    const updateWidth = () => {
      const vw = window.innerWidth
      const mobile = vw <= 768
      setIsMobile(mobile)
      if (mobile) {
        // Mobile: single-page, full width minus padding
        setPageWidth(vw - 32)
      }
      else {
        // Desktop: two-page spread, each half of available width
        const available = Math.min(vw - 120, 1400)
        setPageWidth(Math.floor(available / 2) - 16)
      }
    }
    updateWidth()
    window.addEventListener('resize', updateWidth)
    return () => window.removeEventListener('resize', updateWidth)
  }, [])

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
    <Layout
      title="vLLM Semantic Router"
      description="Signal Driven Decision Routing for Mixture-of-Modality Models — Official White Paper"
    >
      <div className={styles.page}>
        {/* PDF viewer area */}
        <div className={styles.viewerArea}>
          {error ? (
            <div className={styles.fallback}>
              <p>Unable to load PDF preview.</p>
              <a href={PDF_URL} target="_blank" rel="noopener noreferrer">
                Click here to open the PDF in a new tab
              </a>
            </div>
          ) : (
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
            {/* Right: action buttons */}
            <div className={styles.paginationRight}>
              <a
                href={PDF_URL}
                download="vllm-semantic-router.pdf"
                className={`${styles.toolbarBtn} ${styles.toolbarBtnPrimary}`}
              >
                ⬇ Download
              </a>
              <a
                href={PDF_URL}
                target="_blank"
                rel="noopener noreferrer"
                className={`${styles.toolbarBtn} ${styles.toolbarBtnOutline}`}
              >
                ↗ New Tab
              </a>
            </div>
          </div>
        )}
      </div>
    </Layout>
  )
}
