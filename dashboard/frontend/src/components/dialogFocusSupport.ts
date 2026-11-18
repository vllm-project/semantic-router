const FORM_CONTROL_SELECTOR = [
  'input:not([type="hidden"]):not([readonly])',
  'textarea:not([readonly])',
  'select',
  'button[role="combobox"]',
  'button[role="switch"]',
  'button[role="checkbox"]',
]
  .map((selector) => `${selector}:not([disabled]):not([tabindex="-1"]):not([aria-hidden="true"])`)
  .join(',')

function isVisible(element: HTMLElement) {
  return element.getClientRects().length > 0
}

/**
 * Moves focus to the control marked `data-dialog-initial-focus` when it can
 * take focus; otherwise Radix focuses the first tabbable element.
 */
export function focusInitialDialogControl(event: Event, container: HTMLElement | null) {
  const preferred = container?.querySelector<HTMLElement>('[data-dialog-initial-focus]')
  if (!preferred || preferred.matches(':disabled') || !isVisible(preferred)) return
  event.preventDefault()
  preferred.focus({ preventScroll: true })
}

/** Focuses the first editable field of a form dialog instead of its close button. */
export function focusFirstFormControl(event: Event, container: HTMLElement | null) {
  const control = Array.from(
    container?.querySelectorAll<HTMLElement>(FORM_CONTROL_SELECTOR) ?? [],
  ).find(isVisible)
  event.preventDefault()
  ;(control ?? container)?.focus({ preventScroll: true })
}

/**
 * Applies the `returnFocusRef` contract: `always` focuses the given target,
 * `fallback` only does so when the element focused before opening is gone.
 */
export function restoreDialogFocus(
  event: Event,
  previous: Element | null,
  returnTarget: HTMLElement | null | undefined,
  mode: 'fallback' | 'always' = 'fallback',
) {
  if (!returnTarget) return
  if (mode === 'fallback' && previous instanceof HTMLElement && previous.isConnected) return
  event.preventDefault()
  returnTarget.focus({ preventScroll: true })
}

/** Toasts live outside every dialog; interacting with one must not dismiss the dialog. */
export function keepOpenForToastInteraction(event: Event) {
  const target = event.target
  if (target instanceof Element && target.closest('[data-sonner-toaster]')) event.preventDefault()
}
