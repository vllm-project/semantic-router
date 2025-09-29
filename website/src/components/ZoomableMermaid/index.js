import React, { useState, useRef, useEffect, useCallback } from 'react';
import { createPortal } from 'react-dom';
import Mermaid from '@theme/Mermaid';
import styles from './styles.module.css';

const ZoomableMermaid = ({ children, title }) => {
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [isHovered, setIsHovered] = useState(false);
  const modalRef = useRef(null);
  const containerRef = useRef(null);

  const openModal = useCallback(() => {
    setIsModalOpen(true);
    document.body.style.overflow = 'hidden';
  }, []);

  const closeModal = useCallback(() => {
    setIsModalOpen(false);
    document.body.style.overflow = 'unset';
    // Return focus to the original container
    if (containerRef.current) {
      containerRef.current.focus();
    }
  }, []);

  useEffect(() => {
    const handleEscape = (e) => {
      if (e.key === 'Escape' && isModalOpen) {
        closeModal();
      }
    };

    const handleClickOutside = (e) => {
      if (modalRef.current && !modalRef.current.contains(e.target)) {
        closeModal();
      }
    };

    if (isModalOpen) {
      document.addEventListener('keydown', handleEscape);
      document.addEventListener('mousedown', handleClickOutside);
      
      // Focus the modal content when opened
      setTimeout(() => {
        if (modalRef.current) {
          modalRef.current.focus();
        }
      }, 100);
    }

    return () => {
      document.removeEventListener('keydown', handleEscape);
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, [isModalOpen, closeModal]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      document.body.style.overflow = 'unset';
    };
  }, []);

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      openModal();
    }
  };

  const modalContent = (
    <div 
      className={styles.modal}
      role="dialog"
      aria-modal="true"
      aria-labelledby={title ? "modal-title" : undefined}
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
          <button 
            className={styles.closeButton}
            onClick={closeModal}
            aria-label="关闭放大视图"
            type="button"
          >
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <line x1="18" y1="6" x2="6" y2="18"/>
              <line x1="6" y1="6" x2="18" y2="18"/>
            </svg>
          </button>
        </div>
        <div 
          className={styles.modalBody}
          id="modal-description"
          aria-label="放大的 Mermaid 图表"
        >
          <Mermaid value={children} />
        </div>
      </div>
    </div>
  );

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
        aria-label={`点击放大查看 ${title || 'Mermaid 图表'}`}
        aria-expanded={isModalOpen}
      >
        <div className={styles.zoomHint} aria-hidden="true">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <circle cx="11" cy="11" r="8"/>
            <path d="m21 21-4.35-4.35"/>
            <path d="M11 8v6"/>
            <path d="M8 11h6"/>
          </svg>
          <span>点击放大</span>
        </div>
        <Mermaid value={children} />
      </div>

      {isModalOpen && createPortal(modalContent, document.body)}
    </>
  );
};

export default ZoomableMermaid;