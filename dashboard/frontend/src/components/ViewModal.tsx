import React, { useEffect, useRef } from 'react'

import styles from './ViewModal.module.css'
import ViewPanel, { type ViewField, type ViewPanelAction, type ViewSection } from './ViewPanel'

interface ViewModalProps {
  isOpen: boolean
  onClose: () => void
  onEdit?: () => void
  title: string
  sections: ViewSection[]
  actions?: ViewPanelAction[]
  closeLabel?: string
}

const focusableSelector = [
  'a[href]',
  'button:not([disabled])',
  'input:not([disabled])',
  'select:not([disabled])',
  'textarea:not([disabled])',
  '[tabindex]:not([tabindex="-1"])',
].join(',')

export function transitionFromViewToEdit(onClose: () => void, onEdit?: () => void) {
  onClose()
  onEdit?.()
}

const ViewModal: React.FC<ViewModalProps> = ({
  isOpen,
  onClose,
  onEdit,
  title,
  sections,
  actions,
  closeLabel,
}) => {
  const dialogRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!isOpen) return

    const previouslyFocused = document.activeElement instanceof HTMLElement
      ? document.activeElement
      : null
    const previousBodyOverflow = document.body.style.overflow
    document.body.style.overflow = 'hidden'

    const frame = window.requestAnimationFrame(() => {
      const firstControl = dialogRef.current?.querySelector<HTMLElement>(focusableSelector)
      ;(firstControl ?? dialogRef.current)?.focus()
    })

    return () => {
      window.cancelAnimationFrame(frame)
      document.body.style.overflow = previousBodyOverflow
      previouslyFocused?.focus()
    }
  }, [isOpen])

  if (!isOpen) return null

  const handleEdit = onEdit
    ? () => transitionFromViewToEdit(onClose, onEdit)
    : undefined

  const handleDialogKeyDown = (event: React.KeyboardEvent<HTMLDivElement>) => {
    if (event.key === 'Escape') {
      event.preventDefault()
      onClose()
      return
    }

    if (event.key !== 'Tab' || !dialogRef.current) return

    const controls = Array.from(
      dialogRef.current.querySelectorAll<HTMLElement>(focusableSelector),
    ).filter((element) => element.getAttribute('aria-hidden') !== 'true')

    if (controls.length === 0) {
      event.preventDefault()
      dialogRef.current.focus()
      return
    }

    const firstControl = controls[0]
    const lastControl = controls[controls.length - 1]
    const activeElement = document.activeElement

    if (
      event.shiftKey
      && (activeElement === firstControl || !dialogRef.current.contains(activeElement))
    ) {
      event.preventDefault()
      lastControl.focus()
    } else if (!event.shiftKey && activeElement === lastControl) {
      event.preventDefault()
      firstControl.focus()
    }
  }

  return (
    <div className={styles.overlay} onClick={onClose}>
      <div
        ref={dialogRef}
        className={styles.drawerShell}
        role="dialog"
        aria-modal="true"
        aria-label={title}
        tabIndex={-1}
        onKeyDown={handleDialogKeyDown}
        onClick={(event) => event.stopPropagation()}
      >
        <ViewPanel
          title={title}
          sections={sections}
          onClose={onClose}
          onEdit={handleEdit}
          actions={actions}
          closeLabel={closeLabel}
        />
      </div>
    </div>
  )
}

export type { ViewField, ViewPanelAction, ViewSection }
export default ViewModal
