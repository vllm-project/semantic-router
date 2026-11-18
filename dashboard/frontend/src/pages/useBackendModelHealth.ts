import { useCallback, useEffect, useRef, useState } from 'react'

import {
  fetchModelHealth,
  ModelHealthApiError,
  type ModelHealth,
  type ModelHealthSnapshot,
} from '../utils/modelHealthApi'

export interface BackendModelHealth {
  /** unavailable: the Dashboard has no health checks, or this viewer may not read them. */
  phase: 'loading' | 'ready' | 'unavailable'
  byModel: ReadonlyMap<string, ModelHealth>
  checkedAt?: string
  intervalSeconds: number
  checking: boolean
  checkingModels: ReadonlySet<string>
}

const DEFAULT_INTERVAL_SECONDS = 30
// An older Dashboard without health checks, or a viewer who may not read
// them, answers the same way on every later poll.
const PERMANENT_STATUSES = new Set([401, 403, 404, 405, 501])

const INITIAL_HEALTH: BackendModelHealth = {
  phase: 'loading',
  byModel: new Map(),
  intervalSeconds: DEFAULT_INTERVAL_SECONDS,
  checking: false,
  checkingModels: new Set(),
}

/** A full snapshot replaces the models; a snapshot of some models updates only them. */
export function mergeModelHealth(
  current: ReadonlyMap<string, ModelHealth>,
  snapshot: ModelHealthSnapshot,
  partial: boolean,
): Map<string, ModelHealth> {
  const next = new Map(partial ? current : [])
  for (const entry of snapshot.models) next.set(entry.model, entry)
  return next
}

export function isPermanentHealthFailure(error: unknown): boolean {
  return error instanceof ModelHealthApiError && PERMANENT_STATUSES.has(error.status)
}

/**
 * Backend model health, checked on load, after every configuration change
 * and every interval while the page is visible, with on-demand checks of all
 * models or one.
 */
export function useBackendModelHealth(configRevision: unknown) {
  const [state, setState] = useState<BackendModelHealth>(INITIAL_HEALTH)
  const interval = useRef(DEFAULT_INTERVAL_SECONDS)

  const load = useCallback(
    async (options: { refresh?: boolean; models?: string[]; signal?: AbortSignal }) => {
      const snapshot = await fetchModelHealth(options)
      const partial = Boolean(options.models?.length)
      interval.current = snapshot.intervalSeconds
      setState((current) => ({
        ...current,
        phase: 'ready',
        byModel: mergeModelHealth(current.byModel, snapshot, partial),
        checkedAt: partial ? (current.checkedAt ?? snapshot.checkedAt) : snapshot.checkedAt,
        intervalSeconds: snapshot.intervalSeconds,
      }))
      return snapshot
    },
    [],
  )

  useEffect(() => {
    const controller = new AbortController()
    let timeout: number | undefined
    let loading = false
    let stopped = false
    let lastLoaded = 0
    const schedule = () => {
      window.clearTimeout(timeout)
      if (!stopped && !controller.signal.aborted) {
        timeout = window.setTimeout(() => void tick(), interval.current * 1000)
      }
    }
    const tick = async () => {
      if (loading) return
      if (document.visibilityState === 'hidden') {
        schedule()
        return
      }
      loading = true
      lastLoaded = Date.now()
      try {
        await load({ signal: controller.signal })
      } catch (error) {
        if (controller.signal.aborted) return
        stopped = isPermanentHealthFailure(error)
        // A passing failure keeps the last answer; the next poll may succeed.
        setState((current) =>
          current.phase === 'ready' && !stopped ? current : { ...current, phase: 'unavailable' },
        )
      } finally {
        loading = false
      }
      schedule()
    }
    const onVisibilityChange = () => {
      if (
        document.visibilityState === 'visible' &&
        !stopped &&
        Date.now() - lastLoaded >= interval.current * 1000
      ) {
        void tick()
      }
    }
    void tick()
    document.addEventListener('visibilitychange', onVisibilityChange)
    return () => {
      controller.abort()
      window.clearTimeout(timeout)
      document.removeEventListener('visibilitychange', onVisibilityChange)
    }
  }, [configRevision, load])

  const checkAll = useCallback(async () => {
    setState((current) => ({ ...current, checking: true }))
    try {
      return await load({ refresh: true })
    } finally {
      setState((current) => ({ ...current, checking: false }))
    }
  }, [load])

  const checkModel = useCallback(
    async (model: string) => {
      setState((current) => ({
        ...current,
        checkingModels: new Set(current.checkingModels).add(model),
      }))
      try {
        const snapshot = await load({ refresh: true, models: [model] })
        return snapshot.models.find((entry) => entry.model === model)
      } finally {
        setState((current) => {
          const checkingModels = new Set(current.checkingModels)
          checkingModels.delete(model)
          return { ...current, checkingModels }
        })
      }
    },
    [load],
  )

  return { ...state, checkAll, checkModel }
}

export type BackendModelHealthController = ReturnType<typeof useBackendModelHealth>
