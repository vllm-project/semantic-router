import { type FormEvent, type ReactNode, type RefObject, useEffect, useId, useState } from 'react'

import useAccessibleDialog from '../hooks/useAccessibleDialog'
import styles from './ConfirmDialog.module.css'

interface ConfirmDialogProps {
  isOpen: boolean
  title: string
  description: ReactNode
  confirmLabel?: string
  cancelLabel?: string
  eyebrow?: string
  details?: ReactNode
  errorMessage?: ReactNode
  errorDetails?: ReactNode
  pending?: boolean
  pendingLabel?: string
  tone?: 'danger' | 'warning' | 'neutral'
  confirmationText?: string
  returnFocusRef?: RefObject<HTMLElement | null>
  returnFocusMode?: 'fallback' | 'always'
  onCancel: () => void
  onConfirm: () => void | Promise<void>
}

interface ConfirmDialogFormProps {
  confirmation: string
  confirmationText?: string
  confirmationReady: boolean
  confirmLabel: string
  cancelLabel: string
  pending: boolean
  pendingLabel: string
  tone: NonNullable<ConfirmDialogProps['tone']>
  onCancel: () => void
  onConfirmationChange: (value: string) => void
  onSubmit: (event: FormEvent) => void
}

function ConfirmDialogHeader({
  titleId,
  descriptionId,
  eyebrow,
  title,
  description,
}: Pick<ConfirmDialogProps, 'eyebrow' | 'title' | 'description'> & {
  titleId: string
  descriptionId: string
}) {
  return (
    <div className={styles.copy}>
      <span className={styles.eyebrow}>{eyebrow}</span>
      <h2 id={titleId}>{title}</h2>
      <div id={descriptionId} className={styles.description}>
        {description}
      </div>
    </div>
  )
}

function ConfirmDialogError({
  errorId,
  errorMessage,
  errorDetails,
}: Pick<ConfirmDialogProps, 'errorMessage' | 'errorDetails'> & { errorId: string }) {
  if (!errorMessage) return null
  return (
    <div id={errorId} className={styles.error} role="alert">
      <div>{errorMessage}</div>
      {errorDetails}
    </div>
  )
}

function ConfirmDialogForm({
  confirmation,
  confirmationText,
  confirmationReady,
  confirmLabel,
  cancelLabel,
  pending,
  pendingLabel,
  tone,
  onCancel,
  onConfirmationChange,
  onSubmit,
}: ConfirmDialogFormProps) {
  return (
    <form onSubmit={onSubmit}>
      {confirmationText ? (
        <label className={styles.confirmation}>
          <span className={styles.confirmationInstruction}>
            Enter <strong>{confirmationText}</strong> to confirm.
          </span>
          <input
            type="text"
            value={confirmation}
            onChange={(event) => onConfirmationChange(event.target.value)}
            autoComplete="off"
            data-dialog-initial-focus
          />
        </label>
      ) : null}
      <div className={styles.actions}>
        <button
          type="button"
          className={styles.cancelButton}
          onClick={onCancel}
          disabled={pending}
          data-dialog-initial-focus={!confirmationText ? true : undefined}
        >
          {cancelLabel}
        </button>
        <button
          type="submit"
          className={`${styles.confirmButton} ${styles[tone]}`}
          disabled={pending || !confirmationReady}
        >
          {pending ? pendingLabel : confirmLabel}
        </button>
      </div>
    </form>
  )
}

interface ConfirmDialogPanelProps extends ConfirmDialogFormProps {
  titleId: string
  descriptionId: string
  errorId: string
  title: string
  description: ReactNode
  eyebrow: string
  details?: ReactNode
  errorMessage?: ReactNode
  errorDetails?: ReactNode
  dialogRef: RefObject<HTMLElement>
}

function ConfirmDialogPanel(props: ConfirmDialogPanelProps) {
  const { pending, onCancel, tone, dialogRef, titleId, descriptionId, errorId } = props
  return (
    <div
      className={styles.overlay}
      role="presentation"
      onMouseDown={pending ? undefined : onCancel}
    >
      <section
        ref={dialogRef}
        className={styles.dialog}
        role="alertdialog"
        aria-modal="true"
        aria-labelledby={titleId}
        aria-describedby={`${descriptionId}${props.errorMessage ? ` ${errorId}` : ''}`}
        aria-busy={pending}
        tabIndex={-1}
        onMouseDown={(event) => event.stopPropagation()}
      >
        <div className={`${styles.signal} ${styles[tone]}`} aria-hidden="true">
          {tone === 'neutral' ? 'i' : '!'}
        </div>
        <ConfirmDialogHeader {...props} />
        {props.details ? <div className={styles.details}>{props.details}</div> : null}
        <ConfirmDialogError {...props} />
        <ConfirmDialogForm {...props} />
      </section>
    </div>
  )
}

export default function ConfirmDialog({
  confirmLabel = 'Confirm',
  cancelLabel = 'Cancel',
  eyebrow = 'Confirm action',
  pending = false,
  pendingLabel = 'Working…',
  tone = 'danger',
  ...props
}: ConfirmDialogProps) {
  const titleId = useId()
  const descriptionId = useId()
  const errorId = useId()
  const [confirmation, setConfirmation] = useState('')
  const dialogRef = useAccessibleDialog<HTMLElement>({
    isOpen: props.isOpen,
    onClose: props.onCancel,
    dismissible: !pending,
    returnFocusRef: props.returnFocusRef,
    returnFocusMode: props.returnFocusMode,
  })
  const confirmationReady = !props.confirmationText || confirmation === props.confirmationText
  useEffect(() => {
    if (props.isOpen) setConfirmation('')
  }, [props.isOpen, props.confirmationText])
  if (!props.isOpen) return null
  const handleSubmit = (event: FormEvent) => {
    event.preventDefault()
    if (pending || !confirmationReady) return
    void props.onConfirm()
  }
  return (
    <ConfirmDialogPanel
      {...props}
      titleId={titleId}
      descriptionId={descriptionId}
      errorId={errorId}
      dialogRef={dialogRef}
      confirmation={confirmation}
      confirmationReady={confirmationReady}
      confirmLabel={confirmLabel}
      cancelLabel={cancelLabel}
      eyebrow={eyebrow}
      pending={pending}
      pendingLabel={pendingLabel}
      tone={tone}
      onConfirmationChange={setConfirmation}
      onSubmit={handleSubmit}
    />
  )
}
