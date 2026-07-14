import { useId, type RefObject } from 'react'
import useAccessibleDialog from '../hooks/useAccessibleDialog'
import styles from './DslEditorPage.module.css'

interface DslImportModalProps {
  importText: string
  importError: string | null
  importUrl: string
  importUrlLoading: boolean
  textareaRef: RefObject<HTMLTextAreaElement>
  onClose: () => void
  onUrlChange: (value: string) => void
  onFetchUrl: () => void
  onTextChange: (value: string) => void
  onLoadFile: () => void
  onImport: () => void
}

export function DslImportModal({
  importText,
  importError,
  importUrl,
  importUrlLoading,
  textareaRef,
  onClose,
  onUrlChange,
  onFetchUrl,
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
            Paste a full router config YAML or routing fragment below, load from a file, or fetch
            from a URL. Only the routing section will be decompiled into DSL.
          </p>
          <div className={styles.importUrlRow}>
            <input
              className={styles.importUrlInput}
              type="url"
              aria-label="YAML config URL"
              value={importUrl}
              onChange={(event) => onUrlChange(event.target.value)}
              placeholder="https://example.com/config.yaml"
              onKeyDown={(event) => {
                if (event.key === 'Enter') onFetchUrl()
              }}
            />
            <button
              className={styles.toolbarBtn}
              onClick={onFetchUrl}
              disabled={importUrlLoading || !importUrl.trim()}
            >
              {importUrlLoading ? (
                <>
                  <span className={styles.dotPulse} />
                  Fetching…
                </>
              ) : (
                <>
                  <svg
                    width="12"
                    height="12"
                    viewBox="0 0 16 16"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="1.5"
                  >
                    <path d="M6 2a4 4 0 100 8 4 4 0 000-8z" />
                    <path d="M2 6h8M6 2v8" strokeLinecap="round" />
                    <path d="M14 14l-3.5-3.5" strokeLinecap="round" />
                  </svg>
                  Fetch
                </>
              )}
            </button>
          </div>
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
