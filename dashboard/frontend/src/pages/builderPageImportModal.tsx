import React from "react";

import styles from "./BuilderPage.module.css";

interface BuilderImportModalProps {
  open: boolean;
  importUrl: string;
  importText: string;
  importError: string | null;
  importUrlLoading: boolean;
  loadingFromRouter: boolean;
  importTextareaRef: React.Ref<HTMLTextAreaElement>;
  onClose: () => void;
  onImportUrlChange: (value: string) => void;
  onImportTextChange: (value: string) => void;
  onImportUrl: () => void;
  onSelectFile: () => void;
  onLoadFromRouter: () => void;
  onConfirm: () => void;
}

const BuilderImportModal: React.FC<BuilderImportModalProps> = ({
  open,
  importUrl,
  importText,
  importError,
  importUrlLoading,
  loadingFromRouter,
  importTextareaRef,
  onClose,
  onImportUrlChange,
  onImportTextChange,
  onImportUrl,
  onSelectFile,
  onLoadFromRouter,
  onConfirm,
}) => {
  if (!open) return null;

  return (
    <div className={styles.modalOverlay} onClick={onClose}>
      <div className={styles.modal} onClick={(event) => event.stopPropagation()}>
        <div className={styles.modalHeader}>
          <h3 className={styles.modalTitle}>Import Config</h3>
          <button className={styles.modalClose} onClick={onClose}>
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
            Paste a router config YAML below, load from a file, fetch from a
            URL, or load the current router config directly. It will be
            decompiled into DSL.
          </p>
          <div className={styles.importUrlRow}>
            <input
              className={styles.importUrlInput}
              type="url"
              value={importUrl}
              onChange={(event) => onImportUrlChange(event.target.value)}
              placeholder="https://example.com/config.yaml"
              onKeyDown={(event) => {
                if (event.key === "Enter") onImportUrl();
              }}
            />
            <button
              className={styles.toolbarBtn}
              onClick={onImportUrl}
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
            ref={importTextareaRef}
            className={styles.importTextarea}
            value={importText}
            onChange={(event) => onImportTextChange(event.target.value)}
            placeholder="Paste YAML config here..."
            spellCheck={false}
          />
          {importError && <div className={styles.importError}>{importError}</div>}
        </div>
        <div className={styles.modalFooter}>
          <div className={styles.modalFooterImportActions}>
            <button className={styles.toolbarBtn} onClick={onSelectFile}>
              <svg
                width="12"
                height="12"
                viewBox="0 0 16 16"
                fill="none"
                stroke="currentColor"
                strokeWidth="1.5"
              >
                <path
                  d="M2 14h12M8 2v9M5 5l3-3 3 3"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
              </svg>
              Load File
            </button>
            <button
              className={styles.toolbarBtnPrimary}
              onClick={onLoadFromRouter}
              disabled={loadingFromRouter}
              title="Load the current running router config and decompile to DSL"
            >
              <svg
                width="12"
                height="12"
                viewBox="0 0 16 16"
                fill="none"
                stroke="currentColor"
                strokeWidth="1.5"
              >
                <rect x="2" y="2" width="12" height="12" rx="2" />
                <path
                  d="M8 5v6M5 8l3 3 3-3"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
              </svg>
              {loadingFromRouter ? "Loading…" : "Load from Router"}
            </button>
          </div>
          <div className={styles.modalFooterPrimaryActions}>
            <button className={styles.toolbarBtn} onClick={onClose}>
              Cancel
            </button>
            <button
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
  );
};

export { BuilderImportModal };
