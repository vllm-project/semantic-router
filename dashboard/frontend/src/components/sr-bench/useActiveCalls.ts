import { useEffect, useState } from 'react'
import { benchApi } from './api'
import type { CallRecord } from './types'

interface ActivityRead {
  scope: string
  calls: CallRecord[] | null
  readAt: string | null
  error: string
}

// Active calls are a separate observation from paginated, saved call evidence.
// A long stream need not change the run's completed-result timestamp.
export function useActiveCalls(id: string, enabled: boolean, revision: number) {
  const scope = `${id}:${revision}`
  const [state, setState] = useState<ActivityRead>({
    scope,
    calls: null,
    readAt: null,
    error: '',
  })
  useEffect(() => {
    const controller = new AbortController()
    let timer: ReturnType<typeof setTimeout> | undefined
    setState({ scope, calls: null, readAt: null, error: '' })
    if (!enabled) return () => controller.abort()

    async function read() {
      try {
        const calls = new Map<string, CallRecord>()
        let after = 0
        for (let pageIndex = 0; ; pageIndex++) {
          if (pageIndex >= 10) throw new Error('The active-call page limit was reached.')
          const page = await benchApi.activeCalls(id, after, controller.signal)
          if (controller.signal.aborted) return
          for (const call of page.calls) calls.set(call.id, call)
          if (page.next_cursor === null) break
          if (page.next_cursor <= after) throw new Error('The active-call cursor did not advance.')
          after = page.next_cursor
        }
        setState({ scope, calls: [...calls.values()], readAt: new Date().toISOString(), error: '' })
      } catch (cause) {
        if (controller.signal.aborted) return
        setState((previous) => ({
          ...previous,
          error: cause instanceof Error ? cause.message : 'Activity could not be read.',
        }))
      }
      if (!controller.signal.aborted) timer = setTimeout(() => void read(), 2500)
    }
    void read()
    return () => {
      controller.abort()
      if (timer) clearTimeout(timer)
    }
  }, [id, enabled, scope])
  return state.scope === scope && enabled ? state : { scope, calls: null, readAt: null, error: '' }
}
