import { useEffect, useRef } from 'react'

export interface QuietRefreshOptions {
  /** Poll at this interval while the tab is visible. Leave unset for audited reads. */
  everyMs?: number
  /** Refresh when the tab comes back after being hidden at least this long. */
  afterAwayMs?: number
  enabled?: boolean
}

/**
 * Keeps a list current without a refresh button: polls while the tab is
 * visible and refreshes when a hidden tab comes back.
 */
export function useQuietRefresh(
  refresh: () => void,
  { everyMs, afterAwayMs = 0, enabled = true }: QuietRefreshOptions,
) {
  const latest = useRef(refresh)
  useEffect(() => {
    latest.current = refresh
  })
  useEffect(() => {
    if (!enabled) return undefined
    let hiddenAt = document.hidden ? Date.now() : 0
    const onVisibility = () => {
      if (document.hidden) {
        hiddenAt = Date.now()
        return
      }
      if (hiddenAt && Date.now() - hiddenAt >= afterAwayMs) latest.current()
      hiddenAt = 0
    }
    const timer = everyMs
      ? window.setInterval(() => {
          if (!document.hidden) latest.current()
        }, everyMs)
      : undefined
    document.addEventListener('visibilitychange', onVisibility)
    return () => {
      if (timer !== undefined) window.clearInterval(timer)
      document.removeEventListener('visibilitychange', onVisibility)
    }
  }, [afterAwayMs, enabled, everyMs])
}
