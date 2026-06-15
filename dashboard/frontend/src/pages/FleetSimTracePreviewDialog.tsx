import { useEffect } from 'react'
import { createPortal } from 'react-dom'
import styles from './FleetSimPage.module.css'
import type { TraceInfo, TraceSample } from '../utils/fleetSimApi'
import { formatDateTime, formatNumber, formatTraceFormat } from './fleetSimPageSupport'

interface FleetSimTracePreviewDialogProps {
  trace: TraceInfo | null
  sample: TraceSample | null
  onClose: () => void
}

export default function FleetSimTracePreviewDialog({
  trace,
  sample,
  onClose,
}: FleetSimTracePreviewDialogProps) {
  useEffect(() => {
    if (!trace || !sample) {
      return
    }

    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        onClose()
      }
    }

    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [onClose, sample, trace])

  if (!trace || !sample) {
    return null
  }

  const previewCount = Math.min(sample.records.length, sample.total)

  return createPortal(
    <div className={styles.dialogOverlay} onClick={onClose}>
      <div
        className={styles.dialogPanel}
        onClick={(event) => event.stopPropagation()}
        role="dialog"
        aria-modal="true"
        aria-labelledby="fleet-sim-trace-preview-title"
      >
        <div className={styles.dialogHeader}>
          <div>
            <span className={styles.dialogEyebrow}>Trace Preview</span>
            <h2 id="fleet-sim-trace-preview-title" className={styles.dialogTitle}>
              {trace.name}
            </h2>
            <p className={styles.dialogSubtitle}>
              Showing {formatNumber(previewCount)} of {formatNumber(sample.total)} rows before
              you plan against this traffic.
            </p>
          </div>
          <button
            type="button"
            className={styles.dialogCloseButton}
            aria-label="Close trace preview"
            onClick={onClose}
          >
            ×
          </button>
        </div>

        <div className={styles.dialogMetaRow}>
          <span className={styles.summaryPill}>{formatTraceFormat(trace.format)}</span>
          <span className={styles.summaryPill}>{formatNumber(trace.n_requests)} requests</span>
          <span className={styles.summaryPill}>{formatDateTime(trace.upload_time)}</span>
        </div>

        <div className={styles.dialogBody}>
          {sample.records.length > 0 ? (
            <pre className={styles.dialogJsonPreview}>{JSON.stringify(sample.records, null, 2)}</pre>
          ) : (
            <div className={styles.emptyState}>No preview rows are available for this trace.</div>
          )}
        </div>

        <div className={styles.dialogFooter}>
          <button type="button" className={styles.secondaryButton} onClick={onClose}>
            Close
          </button>
        </div>
      </div>
    </div>,
    document.body
  )
}
