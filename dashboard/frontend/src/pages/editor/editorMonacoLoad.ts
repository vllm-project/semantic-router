export type MonacoLoadState = 'idle' | 'loading' | 'ready' | 'failed'

export type MonacoLoadEvent =
  | { type: 'start' }
  | { type: 'loaded' }
  | { type: 'error'; error: unknown }

/** A load the loader abandoned because its view unmounted is not a failure. */
export function isMonacoCancellation(error: unknown): boolean {
  return (error as { type?: unknown } | null)?.type === 'cancelation'
}

/**
 * The editors' next state after a load event. A failure is final for the page:
 * the plain-text editor and diff stay up rather than switching back to a
 * Monaco whose code or worker has already failed once.
 */
export function nextMonacoLoadState(
  state: MonacoLoadState,
  event: MonacoLoadEvent,
): MonacoLoadState {
  if (state === 'failed') return state
  if (event.type === 'start') return state === 'idle' ? 'loading' : state
  if (event.type === 'loaded') return 'ready'
  return isMonacoCancellation(event.error) ? state : 'failed'
}

export interface MonacoLoad {
  getState: () => MonacoLoadState
  subscribe: (listener: () => void) => () => void
  /** Starts the load on first call; later calls share it. */
  start: () => Promise<void>
  /** Reports a failure outside the load itself: a worker that cannot start, an editor that throws. */
  fail: (error: unknown) => void
}

/** One shared Monaco load. `load` receives `fail` to report failures that surface later. */
export function createMonacoLoad(
  load: (fail: (error: unknown) => void) => Promise<unknown>,
): MonacoLoad {
  let state: MonacoLoadState = 'idle'
  let pending: Promise<void> | null = null
  const listeners = new Set<() => void>()

  const dispatch = (event: MonacoLoadEvent) => {
    const next = nextMonacoLoadState(state, event)
    if (next === state) return
    if (next === 'failed' && event.type === 'error') {
      console.warn('[editor] Monaco is unavailable; using the plain-text editor.', event.error)
    }
    state = next
    listeners.forEach((listener) => listener())
  }
  const fail = (error: unknown) => dispatch({ type: 'error', error })

  return {
    getState: () => state,
    subscribe(listener) {
      listeners.add(listener)
      return () => {
        listeners.delete(listener)
      }
    },
    start() {
      if (!pending) {
        dispatch({ type: 'start' })
        pending = Promise.resolve()
          .then(() => load(fail))
          .then(() => dispatch({ type: 'loaded' }), fail)
      }
      return pending
    },
    fail,
  }
}
