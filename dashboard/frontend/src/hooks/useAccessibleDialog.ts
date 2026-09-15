import { type RefObject, useEffect, useRef } from 'react'

const FOCUSABLE_SELECTOR = [
  'a[href]',
  'area[href]',
  'button:not([disabled])',
  'input:not([disabled]):not([type="hidden"])',
  'select:not([disabled])',
  'textarea:not([disabled])',
  'iframe',
  'object',
  'embed',
  '[contenteditable="true"]',
  '[tabindex]:not([tabindex="-1"])',
].join(',')

let bodyScrollLockCount = 0
let bodyOverflowBeforeLock = ''
const activeDialogStack: symbol[] = []

function isTopmostDialog(dialogKey: symbol) {
  return activeDialogStack[activeDialogStack.length - 1] === dialogKey
}

function acquireBodyScrollLock() {
  if (bodyScrollLockCount === 0) {
    bodyOverflowBeforeLock = document.body.style.overflow
    document.body.style.overflow = 'hidden'
  }
  bodyScrollLockCount += 1

  return () => {
    bodyScrollLockCount = Math.max(0, bodyScrollLockCount - 1)
    if (bodyScrollLockCount === 0) {
      document.body.style.overflow = bodyOverflowBeforeLock
      bodyOverflowBeforeLock = ''
    }
  }
}

function isFocusable(element: HTMLElement) {
  if (element.matches(':disabled') || element.closest('[hidden], [aria-hidden="true"], [inert]')) {
    return false
  }
  return element.offsetWidth > 0 || element.offsetHeight > 0 || element.getClientRects().length > 0
}

interface AccessibleDialogOptions {
  isOpen: boolean
  onClose: () => void
  dismissible?: boolean
  lockBodyScroll?: boolean
  returnFocusRef?: RefObject<HTMLElement | null>
  returnFocusMode?: 'fallback' | 'always'
}

function focusableDialogElements<T extends HTMLElement>(dialogRef: RefObject<T | null>) {
  return Array.from(
    dialogRef.current?.querySelectorAll<HTMLElement>(FOCUSABLE_SELECTOR) ?? [],
  ).filter(isFocusable)
}

function focusInitialDialogControl<T extends HTMLElement>(
  dialogRef: RefObject<T | null>,
  dialogKey: symbol,
) {
  if (!isTopmostDialog(dialogKey)) return
  const dialog = dialogRef.current
  if (!dialog) return
  const preferredControl = dialog.querySelector<HTMLElement>('[data-dialog-initial-focus]')
  const firstControl = focusableDialogElements(dialogRef)[0]
  const target =
    preferredControl && isFocusable(preferredControl) ? preferredControl : (firstControl ?? dialog)
  target.focus({ preventScroll: true })
}

function trapDialogFocus<T extends HTMLElement>(
  dialogRef: RefObject<T | null>,
  event: KeyboardEvent,
) {
  const dialog = dialogRef.current
  if (!dialog) return
  const controls = focusableDialogElements(dialogRef)
  if (controls.length === 0) {
    event.preventDefault()
    dialog.focus({ preventScroll: true })
    return
  }

  const firstControl = controls[0]
  const lastControl = controls[controls.length - 1]
  const activeControlIndex = controls.indexOf(document.activeElement as HTMLElement)
  if (activeControlIndex === -1) {
    event.preventDefault()
    ;(event.shiftKey ? lastControl : firstControl).focus()
  } else if (event.shiftKey && activeControlIndex === 0) {
    event.preventDefault()
    lastControl.focus()
  } else if (!event.shiftKey && activeControlIndex === controls.length - 1) {
    event.preventDefault()
    firstControl.focus()
  }
}

interface ActiveDialogOptions<T extends HTMLElement> {
  dialogRef: RefObject<T | null>
  dialogKey: symbol
  explicitReturnTarget: HTMLElement | null | undefined
  lockBodyScroll: boolean
  dismissible: () => boolean
  close: () => void
  returnFocusMode: () => 'fallback' | 'always'
}

function activateDialog<T extends HTMLElement>({
  dialogRef,
  dialogKey,
  explicitReturnTarget,
  lockBodyScroll,
  dismissible,
  close,
  returnFocusMode,
}: ActiveDialogOptions<T>) {
  const previouslyFocused =
    document.activeElement instanceof HTMLElement ? document.activeElement : null
  const releaseBodyScrollLock = lockBodyScroll ? acquireBodyScrollLock() : undefined
  activeDialogStack.push(dialogKey)
  const frame = window.requestAnimationFrame(() => focusInitialDialogControl(dialogRef, dialogKey))
  const handleKeyDown = (event: KeyboardEvent) => {
    if (!isTopmostDialog(dialogKey)) return
    if (event.key === 'Escape' && dismissible()) {
      event.preventDefault()
      event.stopPropagation()
      close()
      return
    }
    if (event.key === 'Tab') trapDialogFocus(dialogRef, event)
  }

  document.addEventListener('keydown', handleKeyDown, true)
  return () => {
    window.cancelAnimationFrame(frame)
    document.removeEventListener('keydown', handleKeyDown, true)
    const wasTopmost = isTopmostDialog(dialogKey)
    const stackIndex = activeDialogStack.lastIndexOf(dialogKey)
    if (stackIndex >= 0) activeDialogStack.splice(stackIndex, 1)
    releaseBodyScrollLock?.()
    if (!wasTopmost) return
    const returnTarget =
      returnFocusMode() === 'always'
        ? explicitReturnTarget
        : previouslyFocused?.isConnected
          ? previouslyFocused
          : explicitReturnTarget
    returnTarget?.focus({ preventScroll: true })
  }
}

/**
 * Supplies the keyboard and focus behavior required by an aria-modal dialog.
 * Mark a preferred initial control with `data-dialog-initial-focus`; otherwise
 * the first enabled control (or the dialog itself) receives focus.
 */
export default function useAccessibleDialog<T extends HTMLElement>({
  isOpen,
  onClose,
  dismissible = true,
  lockBodyScroll = true,
  returnFocusRef,
  returnFocusMode = 'fallback',
}: AccessibleDialogOptions) {
  const dialogRef = useRef<T>(null)
  const dialogKeyRef = useRef(Symbol('accessible-dialog'))
  const onCloseRef = useRef(onClose)
  const dismissibleRef = useRef(dismissible)
  const returnFocusModeRef = useRef(returnFocusMode)

  onCloseRef.current = onClose
  dismissibleRef.current = dismissible
  returnFocusModeRef.current = returnFocusMode

  useEffect(() => {
    if (!isOpen) return
    return activateDialog({
      dialogRef,
      dialogKey: dialogKeyRef.current,
      explicitReturnTarget: returnFocusRef?.current,
      lockBodyScroll,
      dismissible: () => dismissibleRef.current,
      close: () => onCloseRef.current(),
      returnFocusMode: () => returnFocusModeRef.current,
    })
  }, [isOpen, lockBodyScroll, returnFocusRef])

  return dialogRef
}
