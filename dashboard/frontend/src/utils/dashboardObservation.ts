export interface Observation<T> {
  data: T | null
  error: string | null
  updatedAt: number | null
  loading: boolean
  refreshing: boolean
  stale: boolean
}

export interface ObservationOptions {
  freshMs?: number
  pollMs?: number
  isHidden?: () => boolean
}

// Observations are shared by active consumers. Polls start after the previous
// read settles, so a slow response is never aborted by the next poll tick.
export function createDashboardObservation<T>(
  load: (signal: AbortSignal) => Promise<T>,
  { freshMs = 5_000, pollMs = 10_000, isHidden = () => false }: ObservationOptions = {},
) {
  const empty = (): Observation<T> => ({
    data: null,
    error: null,
    updatedAt: null,
    loading: true,
    refreshing: false,
    stale: false,
  })
  let snapshot = empty()
  const listeners = new Set<() => void>()
  let controller: AbortController | null = null
  let inFlight: Promise<void> | null = null
  let timer: ReturnType<typeof setTimeout> | undefined
  let generation = 0
  let failures = 0
  const publish = (next: Partial<Observation<T>>) => {
    snapshot = { ...snapshot, ...next }
    listeners.forEach((listener) => listener())
  }
  const cancel = () => {
    generation += 1
    clearTimeout(timer)
    controller?.abort()
    controller = null
    inFlight = null
  }
  const schedule = () => {
    clearTimeout(timer)
    if (!listeners.size || (!pollMs && !failures)) return
    const delay = failures ? Math.min(1_000 * 2 ** Math.min(failures, 6), 60_000) : pollMs
    timer = setTimeout(() => {
      if (isHidden()) schedule()
      else void refresh(true)
    }, delay)
  }
  const refresh = (force = true): Promise<void> => {
    if (inFlight) return inFlight
    if (
      !force &&
      !snapshot.stale &&
      snapshot.updatedAt !== null &&
      Date.now() - snapshot.updatedAt < freshMs
    ) {
      schedule()
      return Promise.resolve()
    }
    clearTimeout(timer)
    const current = ++generation
    const request = new AbortController()
    controller = request
    publish({ refreshing: true, loading: snapshot.data === null && !snapshot.error })
    inFlight = Promise.resolve()
      .then(() => {
        request.signal.throwIfAborted()
        return load(request.signal)
      })
      .then((data) => {
        if (current !== generation || request.signal.aborted) return
        failures = 0
        publish({ data, error: null, updatedAt: Date.now(), loading: false, stale: false })
      })
      .catch((error: unknown) => {
        if (current !== generation || request.signal.aborted) return
        failures += 1
        // Keep the last observation visible, explicitly marked as stale. A
        // network error is not evidence that a previously ready model stopped.
        publish({
          error: error instanceof Error ? error.message : 'Observation unavailable.',
          loading: false,
          stale: snapshot.data !== null,
        })
      })
      .finally(() => {
        if (current !== generation) return
        inFlight = null
        controller = null
        publish({ refreshing: false })
        schedule()
      })
    return inFlight
  }
  return {
    getSnapshot: () => snapshot,
    subscribe(listener: () => void) {
      listeners.add(listener)
      if (listeners.size === 1) {
        if (snapshot.updatedAt !== null && Date.now() - snapshot.updatedAt >= freshMs) {
          publish({ stale: true })
        }
        void refresh(false)
      }
      return () => {
        listeners.delete(listener)
        if (!listeners.size) {
          cancel()
          snapshot = { ...snapshot, refreshing: false }
        }
      }
    },
    refresh,
    invalidate() {
      cancel()
      failures = 0
      publish({ stale: snapshot.data !== null, refreshing: false })
      if (listeners.size) void refresh()
    },
    replace(data: T) {
      cancel()
      failures = 0
      publish({
        data,
        error: null,
        updatedAt: Date.now(),
        loading: false,
        refreshing: false,
        stale: false,
      })
      schedule()
    },
    clear() {
      cancel()
      failures = 0
      snapshot = empty()
      listeners.forEach((listener) => listener())
    },
  }
}
