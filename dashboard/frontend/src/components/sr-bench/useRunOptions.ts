import { useCallback, useEffect, useRef, useState } from 'react'
import { benchApi, SrBenchRequestError } from './api'
import type { RunChoice, RunOptions } from './types'

interface PageState {
  scope: string
  items: RunChoice[]
  baseline: RunChoice | null
  next: string | null
  loaded: boolean
  loading: boolean
  error: string
  scanLimited: boolean
  unverifiedPairs: number
}
const empty: Omit<PageState, 'scope'> = {
  items: [],
  baseline: null,
  next: null,
  loaded: false,
  loading: false,
  error: '',
  scanLimited: false,
  unverifiedPairs: 0,
}
const validChoice = (item: RunChoice) =>
  typeof item?.run_id === 'string' &&
  !!item.run_id &&
  typeof item.name === 'string' &&
  typeof item.profile === 'string' &&
  Number.isInteger(item.case_count) &&
  item.case_count >= 0

export default function useRunOptions(
  kind: 'comparison' | 'replay',
  baseline?: string,
  enabled = true,
) {
  const scope = `${kind}:${baseline ?? ''}`
  const [state, setState] = useState<PageState>({ ...empty, scope: '' })
  const sequence = useRef(0)
  const request = useRef<AbortController | null>(null)
  const failedCursor = useRef<string | undefined>()
  const read = useCallback(
    async (after?: string) => {
      request.current?.abort()
      failedCursor.current = after
      const controller = new AbortController()
      request.current = controller
      const currentSequence = ++sequence.current
      const current = () => !controller.signal.aborted && sequence.current === currentSequence
      setState((previous) => ({
        ...(after && previous.scope === scope ? previous : empty),
        scope,
        loading: true,
        error: '',
      }))
      try {
        const value: RunOptions = await benchApi.runOptions(
          kind,
          baseline,
          after,
          controller.signal,
        )
        if (!current()) return
        const items = baseline ? value.options : value.baselines
        if (
          value.model_requests !== 0 ||
          typeof value.scan_limited !== 'boolean' ||
          !Number.isInteger(value.unverified_pairs) ||
          value.unverified_pairs < 0 ||
          !Number.isInteger(value.unverified_baselines) ||
          value.unverified_baselines < 0 ||
          !Array.isArray(items) ||
          items.length > 10 ||
          !items.every(validChoice) ||
          new Set(items.map((item) => item.run_id)).size !== items.length ||
          typeof value.has_more !== 'boolean' ||
          (value.has_more && (!value.next_cursor || value.next_cursor === after)) ||
          (baseline &&
            value.baseline !== null &&
            (!validChoice(value.baseline) || value.baseline.run_id !== baseline)) ||
          (baseline && value.baseline === null && items.length > 0)
        ) {
          throw new Error(
            'The service returned an inconsistent selection page. Refresh available runs.',
          )
        }
        setState((previous) => {
          const merged = new Map(
            (after && previous.scope === scope ? previous.items : []).map((item) => [
              item.run_id,
              item,
            ]),
          )
          items.forEach((item) => merged.set(item.run_id, item))
          return {
            scope,
            items: [...merged.values()],
            baseline: value.baseline,
            next: value.has_more ? value.next_cursor : null,
            loaded: true,
            loading: false,
            error: '',
            scanLimited: value.scan_limited,
            unverifiedPairs: value.unverified_pairs,
          }
        })
      } catch (cause) {
        if (current() && cause instanceof SrBenchRequestError && cause.status === 400)
          failedCursor.current = undefined
        if (current())
          setState((previous) => ({
            ...previous,
            scope,
            loading: false,
            error: cause instanceof Error ? cause.message : 'Available runs could not be loaded.',
          }))
      }
    },
    [kind, baseline, scope],
  )
  useEffect(() => {
    if (enabled) void read()
    const pendingSequence = sequence
    return () => {
      pendingSequence.current++
      request.current?.abort()
    }
  }, [read, enabled])
  const current = enabled && state.scope === scope ? state : { ...empty, scope }
  return {
    ...current,
    loading: enabled && ((!current.loaded && !current.error) || current.loading),
    reload: () => void read(failedCursor.current),
    refresh: () => void read(),
    loadMore: () => {
      if (current.next && !current.loading) void read(current.next)
    },
  }
}
