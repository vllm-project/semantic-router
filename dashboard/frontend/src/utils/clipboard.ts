export async function copyText(text: string): Promise<boolean> {
  if (!text) return false

  if (window.isSecureContext && navigator.clipboard?.writeText) {
    try {
      await navigator.clipboard.writeText(text)
      return true
    } catch {
      // Insecure deployments and restrictive browser policies use the
      // selection-based fallback below.
    }
  }

  const textarea = document.createElement('textarea')
  const previousFocus =
    typeof HTMLElement !== 'undefined' && document.activeElement instanceof HTMLElement
      ? document.activeElement
      : null
  let clipboardEventHandled = false
  const handleCopy = (event: ClipboardEvent) => {
    if (!event.clipboardData) return
    event.clipboardData.setData('text/plain', text)
    event.preventDefault()
    clipboardEventHandled = true
  }
  textarea.value = text
  textarea.readOnly = true
  textarea.style.position = 'fixed'
  textarea.style.inset = '0 auto auto -9999px'
  document.body.appendChild(textarea)
  textarea.focus()
  textarea.select()
  textarea.setSelectionRange(0, textarea.value.length)
  document.addEventListener('copy', handleCopy, { once: true })
  try {
    const commandAccepted =
      typeof document.execCommand === 'function' && document.execCommand('copy')
    return commandAccepted && clipboardEventHandled
  } catch {
    return false
  } finally {
    document.removeEventListener('copy', handleCopy)
    textarea.remove()
    previousFocus?.focus()
  }
}
