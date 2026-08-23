import { useId, type RefObject } from 'react'
import useAccessibleDialog from '../hooks/useAccessibleDialog'
import styles from './DslEditorPage.module.css'

interface DslImportModalProps {
  importText: string
  importError: string | null
  textareaRef: RefObject<HTMLTextAreaElement>
  onClose: () => void
  onTextChange: (value: string) => void
  onLoadFile: () => void
  onImport: () => void
}

export function DslImportModal({
  importText,
  importError,
  textareaRef,
  onClose,
  onTextChange,
  onLoadFile,
  onImport,
}: DslImportModalProps) {
  const dialogId = useId()
  const titleId = `${dialogId}-title`
  const dialogRef = useAccessibleDialog<HTMLDivElement>({ isOpen: true, onClose })

  return (
    <div className={styles.modalOverlay} onClick={onClose}>
      <div
        ref={dialogRef}
        id={dialogId}
        className={styles.modal}
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
        tabIndex={-1}
        onClick={(event) => event.stopPropagation()}
      >
        <div className={styles.modalHeader}>
          <h3 id={titleId} className={styles.modalTitle}>
            Import YAML Config
          </h3>
          <button
            className={styles.modalClose}
            aria-label="Close import dialog"
            onClick={onClose}
            data-dialog-initial-focus
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
          <p className={styles.modalHint}>
            Paste a full router config or routing fragment, or load one from a file. Only the
            routing section is imported.
          </p>
          <textarea
            ref={textareaRef}
            className={styles.importTextarea}
            aria-label="YAML config"
            value={importText}
            onChange={(event) => onTextChange(event.target.value)}
            placeholder="Paste YAML config here..."
            spellCheck={false}
          />
          {importError && (
            <div className={styles.importError} role="alert">
              {importError}
            </div>
          )}
        </div>
        <div className={styles.modalFooter}>
          <button className={styles.toolbarBtn} onClick={onLoadFile}>
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
          <div style={{ marginLeft: 'auto', display: 'flex', gap: 'var(--spacing-sm)' }}>
            <button className={styles.toolbarBtn} onClick={onClose}>
              Cancel
            </button>
            <button
              className={styles.toolbarBtnPrimary}
              onClick={onImport}
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
