/** Poll interval while the composer waits for its first registered target. */
export const EMPTY_TARGETS_POLL_MS = 3000

type FocusSource = Pick<EventTarget, 'addEventListener' | 'removeEventListener'>

/**
 * Refetches targets while the evaluation composer is open: whenever the window
 * regains focus (after registering a target from a terminal) and, while no
 * target is registered, every `pollMs`. Returns a cleanup function.
 */
export function watchTargets(
  source: FocusSource,
  refetch: () => void,
  pollMs: number | null,
): () => void {
  source.addEventListener('focus', refetch)
  const timer = pollMs ? setInterval(refetch, pollMs) : undefined
  return () => {
    source.removeEventListener('focus', refetch)
    if (timer) clearInterval(timer)
  }
}
