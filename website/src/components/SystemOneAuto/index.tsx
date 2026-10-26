import React, { useRef } from 'react'
import { AutoResultsFigure } from './AutoResultsFigure'
import { AutoCascadeFigure } from './AutoCascadeFigure'
import { ModelEcosystemsFigure } from './ModelEcosystemsFigure'
import styles from './styles.module.css'

function LandscapeFigure({ title, children }: { title: string, children: React.ReactNode }) {
  const dialog = useRef<HTMLDialogElement>(null)

  return (
    <figure className={styles.figure} aria-label={title}>
      <button
        type="button"
        className={styles.preview}
        aria-label={`Enlarge figure: ${title}`}
        onClick={() => dialog.current?.showModal()}
      >
        {children}
      </button>
      <figcaption className={styles.caption}>
        <span>{title}</span>
        <button type="button" onClick={() => dialog.current?.showModal()}>
          View full size ↗
        </button>
      </figcaption>
      <dialog
        ref={dialog}
        className={styles.dialog}
        aria-label={title}
      >
        <div className={styles.dialogHeader}>
          <strong>{title}</strong>
          <button type="button" onClick={() => dialog.current?.close()}>Close ×</button>
        </div>
        <div className={styles.zoomViewport}>
          <div className={styles.zoomCanvas}>{children}</div>
        </div>
      </dialog>
    </figure>
  )
}

export function AutoResults() {
  return <LandscapeFigure title="Quality, estimated cost and latency"><AutoResultsFigure /></LandscapeFigure>
}

export function AutoCascade() {
  return <LandscapeFigure title="Answer first. Escalate when needed."><AutoCascadeFigure /></LandscapeFigure>
}

export function ModelEcosystems() {
  return <LandscapeFigure title="LLM routing and Decision Model routing"><ModelEcosystemsFigure /></LandscapeFigure>
}
