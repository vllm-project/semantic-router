export const AUTH_SESSION_REVALIDATION_INTERVAL_MS = 5_000

export function subscribeToAuthSessionRevalidation(
  revalidate: () => Promise<void>,
  browserWindow: Window = window,
  browserDocument: Document = document,
): () => void {
  let inFlight = false
  const run = () => {
    if (inFlight) return
    inFlight = true
    void revalidate().finally(() => {
      inFlight = false
    })
  }
  const runWhenVisible = () => {
    if (!browserDocument.hidden) run()
  }
  const interval = browserWindow.setInterval(runWhenVisible, AUTH_SESSION_REVALIDATION_INTERVAL_MS)
  browserWindow.addEventListener('focus', run)
  browserDocument.addEventListener('visibilitychange', runWhenVisible)
  return () => {
    browserWindow.clearInterval(interval)
    browserWindow.removeEventListener('focus', run)
    browserDocument.removeEventListener('visibilitychange', runWhenVisible)
  }
}
