import React, { useId } from 'react'

import useAccessibleDialog from '@/hooks/useAccessibleDialog'

import styles from './BuilderPage.module.css'

interface BuilderImportModalProps {
  open: boolean
  importText: string
  importError: string | null
  importTextareaRef: React.Ref<HTMLTextAreaElement>
  onClose: () => void
  onImportTextChange: (value: string) => void
  onSelectFile: () => void
  onConfirm: () => void
}

const BuilderImportModal: React.FC<BuilderImportModalProps> = ({
  open,
  importText,
  importError,
  importTextareaRef,
  onClose,
  onImportTextChange,
  onSelectFile,
  onConfirm,
}) => {
  const dialogId = useId()
  const titleId = `${dialogId}-title`
  const descriptionId = `${dialogId}-description`
  const dialogRef = useAccessibleDialog<HTMLDivElement>({
    isOpen: open,
    onClose,
  })

  if (!open) return null

  return (
    <div className={styles.modalOverlay} role="presentation" onMouseDown={onClose}>
      <div
        ref={dialogRef}
        className={styles.modal}
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
        aria-describedby={descriptionId}
        tabIndex={-1}
        onMouseDown={(event) => event.stopPropagation()}
      >
        <div className={styles.modalHeader}>
          <h3 id={titleId} className={styles.modalTitle}>
            Import Recipe
          </h3>
          <button
            type="button"
            className={styles.modalClose}
            onClick={onClose}
            aria-label="Close import dialog"
          >
            <svg
              width="14"
              height="14"
              viewBox="0 0 16 16"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
            >
              <path d="M4 4l8 8M12 4l-8 8" strokeLinecap="round" />
            </svg>
          </button>
        </div>
        <div className={styles.modalBody}>
          <p id={descriptionId} className={styles.modalHint}>
            Replace the current draft with one Recipe from a file or pasted document. Models and
            Entrypoints are ignored.
          </p>
          <textarea
            ref={importTextareaRef}
            className={styles.importTextarea}
            aria-label="Recipe document"
            value={importText}
            onChange={(event) => onImportTextChange(event.target.value)}
            placeholder="Paste a Recipe document…"
            spellCheck={false}
            data-dialog-initial-focus
          />
          {importError && (
            <div className={styles.importError} role="alert">
              {importError}
            </div>
          )}
        </div>
        <div className={styles.modalFooter}>
          <div className={styles.modalFooterImportActions}>
            <button type="button" className={styles.toolbarBtn} onClick={onSelectFile}>
              <svg
                width="12"
                height="12"
                viewBox="0 0 16 16"
                fill="none"
                stroke="currentColor"
                strokeWidth="1.5"
              >
                <path d="M2 14h12M8 2v9M5 5l3-3 3 3" strokeLinecap="round" strokeLinejoin="round" />
              </svg>
              Load File
            </button>
          </div>
          <div className={styles.modalFooterPrimaryActions}>
            <button type="button" className={styles.toolbarBtn} onClick={onClose}>
              Cancel
            </button>
            <button
              type="button"
              className={styles.toolbarBtnPrimary}
              onClick={onConfirm}
              disabled={!importText.trim()}
            >
              Import
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}

export { BuilderImportModal }
