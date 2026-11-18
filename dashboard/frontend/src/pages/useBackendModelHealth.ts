import { useCallback, useEffect, useRef, useState } from 'react'

import {
  fetchModelHealth,
  modelHealthIssue,
  type ModelHealth,
  type ModelHealthIssue,
  type ModelHealthSnapshot,
} from '../utils/modelHealthApi'

/**
 * loading: the first check is in flight.
 * ready: an answer arrived; it stays while a later check fails in passing.
 * retrying: no answer yet, and the last check failed in passing; polling continues.
 * unavailable: this account cannot read health, or this Dashboard has none; polling stops.
 */
export type ModelHealthPhase = 'loading' | 'ready' | 'retrying' | 'unavailable'

export interface BackendModelHealth {
  phase: ModelHealthPhase
  /** Why the last check failed; the next answer clears it. */
  issue?: ModelHealthIssue
  byModel: ReadonlyMap<string, ModelHealth>
  checkedAt?: string
  intervalSeconds: number
  checking: boolean
  checkingModels: ReadonlySet<string>
}

const DEFAULT_INTERVAL_SECONDS = 30

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
  return modelHealthIssue(error) !== 'unreachable'
}

/**
 * The state after a failed poll. A refusal or a missing endpoint stops
 * polling; a passing failure keeps the last answer and polls again.
 */
export function healthAfterFailure(
  current: BackendModelHealth,
  error: unknown,
): { state: BackendModelHealth; stop: boolean } {
  const issue = modelHealthIssue(error)
  if (issue !== 'unreachable')
    return { state: { ...current, phase: 'unavailable', issue }, stop: true }
  return {
    state: { ...current, phase: current.phase === 'ready' ? 'ready' : 'retrying', issue },
    stop: false,
  }
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
        issue: undefined,
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
        setState((current) => healthAfterFailure(current, error).state)
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
