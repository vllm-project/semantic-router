const NESTED_POPUP = '[role="listbox"], [role="menu"]'

/**
 * Whether an Escape belongs to a popup inside a dialog or sheet, such as a
 * select list or a menu. A select list unmounts as it handles the key, so
 * by the time the dialog sees the event its target can be detached.
 */
export function escapeBelongsToNestedPopup(event: Pick<KeyboardEvent, 'target'>): boolean {
  const target = event.target as {
    isConnected?: boolean
    closest?: (selector: string) => unknown
  } | null
  if (!target || typeof target.closest !== 'function') return false
  return target.isConnected === false || Boolean(target.closest(NESTED_POPUP))
}
