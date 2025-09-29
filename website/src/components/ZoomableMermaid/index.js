import React, { useState, useRef, useEffect, useCallback } from 'react'
import { createPortal } from 'react-dom'
import Mermaid from '@theme/Mermaid'
import styles from './styles.module.css'

const ZoomableMermaid = ({ children, title, defaultZoom = 1.2 }) => {
  const [isModalOpen, setIsModalOpen] = useState(false)
  const [isHovered, setIsHovered] = useState(false)
  const [zoomLevel, setZoomLevel] = useState(defaultZoom) // Use defaultZoom prop
  const modalRef = useRef(null)
  const containerRef = useRef(null)

  const openModal = useCallback(() => {
    setIsModalOpen(true)
    setZoomLevel(defaultZoom) // Reset to default zoom when opening
    document.body.style.overflow = 'hidden'
  }, [defaultZoom])

  const closeModal = useCallback(() => {
    setIsModalOpen(false)
    document.body.style.overflow = 'unset'
    // Return focus to the original container
    if (containerRef.current) {
      containerRef.current.focus()
    }
  }, [])

  const zoomIn = useCallback(() => {
    setZoomLevel(prev => Math.min(prev + 0.2, 5.0)) // Max 500%
  }, [])

  const zoomOut = useCallback(() => {
    setZoomLevel(prev => Math.max(prev - 0.2, 0.5)) // Min 50%
  }, [])

  const resetZoom = useCallback(() => {
    setZoomLevel(defaultZoom) // Reset to custom default instead of hardcoded 1.2
  }, [defaultZoom])

  useEffect(() => {
    const handleEscape = (e) => {
      if (e.key === 'Escape' && isModalOpen) {
        closeModal()
      }
    }

    const handleClickOutside = (e) => {
      if (modalRef.current && !modalRef.current.contains(e.target)) {
        closeModal()
      }
    }

    const handleKeydown = (e) => {
      if (!isModalOpen) return

      if (e.key === '=' || e.key === '+') {
        e.preventDefault()
        zoomIn()
      }
      else if (e.key === '-') {
        e.preventDefault()
        zoomOut()
      }
      else if (e.key === '0') {
        e.preventDefault()
        resetZoom()
      }
    }

    if (isModalOpen) {
      document.addEventListener('keydown', handleEscape)
      document.addEventListener('mousedown', handleClickOutside)
      document.addEventListener('keydown', handleKeydown)

      // Focus the modal content when opened
      setTimeout(() => {
        if (modalRef.current) {
          modalRef.current.focus()
        }
      }, 100)
    }

    return () => {
      document.removeEventListener('keydown', handleEscape)
      document.removeEventListener('mousedown', handleClickOutside)
      document.removeEventListener('keydown', handleKeydown)
    }
  }, [isModalOpen, closeModal, zoomIn, zoomOut, resetZoom])

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      document.body.style.overflow = 'unset'
    }
  }, [])

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault()
      openModal()
    }
  }

  const modalContent = (
    <div
      className={styles.modal}
      role="dialog"
      aria-modal="true"
      aria-labelledby={title ? 'modal-title' : undefined}
      aria-describedby="modal-description"
    >
      <div
        className={styles.modalContent}
        ref={modalRef}
        tabIndex={-1}
      >
        <div className={styles.modalHeader}>
          {title && (
            <h3 id="modal-title" className={styles.modalTitle}>
              {title}
            </h3>
          )}
          <div className={styles.modalControls}>
            <span className={styles.zoomIndicator}>
              {Math.round(zoomLevel * 100)}
              %
            </span>
            <button
              className={styles.zoomButton}
              onClick={zoomOut}
              disabled={zoomLevel <= 0.5}
              aria-label="Reduce the size of the chart"
              type="button"
              title="Reduce (Shortcut key: -)"
            >
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <circle cx="11" cy="11" r="8" />
                <path d="M8 11h6" />
                <path d="m21 21-4.35-4.35" />
              </svg>
            </button>
            <button
              className={styles.resetButton}
              onClick={resetZoom}
              aria-label={`Reset to default zoom level ${Math.round(defaultZoom * 100)}%`}
              type="button"
              title={`Reset to default zoom level ${Math.round(defaultZoom * 100)}% (Shortcut key: 0)`}
            >
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M3 3l18 18" />
                <path d="m19 4-7 7-7-7" />
                <path d="m5 20 7-7 7 7" />
              </svg>
            </button>
            <button
              className={styles.zoomButton}
              onClick={zoomIn}
              disabled={zoomLevel >= 5.0}
              aria-label="Enlarge the chart"
              type="button"
              title="Enlarge (Shortcut key: +)"
            >
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <circle cx="11" cy="11" r="8" />
                <path d="M8 11h6" />
                <path d="M11 8v6" />
                <path d="m21 21-4.35-4.35" />
              </svg>
            </button>
            <button
              className={styles.closeButton}
              onClick={closeModal}
              aria-label="Close the zoomed view"
              type="button"
            >
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <line x1="18" y1="6" x2="6" y2="18" />
                <line x1="6" y1="6" x2="18" y2="18" />
              </svg>
            </button>
          </div>
        </div>
        <div
          className={styles.modalBody}
          id="modal-description"
          aria-label="Enlarged Mermaid diagram"
        >
          <div
            className={styles.diagramContainer}
            style={{
              transform: `scale(${zoomLevel})`,
              // Ensure scaling is from the center of the diagram.
              // Fix the issue where the top scroll bar is not visible when the chart is enlarged.
              transformOrigin: 'center top',
            }}
          >
            <Mermaid value={children} />
          </div>
        </div>
      </div>
    </div>
  )

  return (
    <>
      <div
        ref={containerRef}
        className={`${styles.mermaidContainer} ${isHovered ? styles.hovered : ''}`}
        onClick={openModal}
        onMouseEnter={() => setIsHovered(true)}
        onMouseLeave={() => setIsHovered(false)}
        role="button"
        tabIndex={0}
        onKeyDown={handleKeyDown}
        aria-label={`Click to enlarge ${title || 'Mermaid diagram'}`}
        aria-expanded={isModalOpen}
      >
        <div className={styles.zoomHint} aria-hidden="true">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <circle cx="11" cy="11" r="8" />
            <path d="m21 21-4.35-4.35" />
            <path d="M11 8v6" />
            <path d="M8 11h6" />
          </svg>
          <span>Click to enlarge</span>
        </div>
        <Mermaid value={children} />
      </div>

      {isModalOpen && createPortal(modalContent, document.body)}
    </>
  )
}

export default ZoomableMermaid
