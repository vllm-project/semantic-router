import { useCallback, useEffect, useState } from 'react'

import {
  chosenSession,
  displaySessionId,
  isScopedSessionId,
  sessionParam,
} from './insightsAgentIdentity'
import { fetchInsightsJSON } from './insightsPageApi'
import type { RunListResponse, RunSummary } from './insightsRunTypes'

const RUNS = '/api/router/api/v1/observability/replays/runs'
/** The sessions one lookup reads: two share an id only when their callers chose the same one. */
const LOOKUP_LIMIT = 20

/** Lookups that found sessions, shared by every view that names the same id. */
const lookups = new Map<string, Promise<RunSummary[]>>()

/** The sessions a harness id names: the ones whose id, without its key scope, is that id. */
export function sessionMatches(raw: string, runs: RunSummary[]): RunSummary[] {
  return runs.filter((run) => displaySessionId(run.session_id) === raw)
}

/** The sessions the signed-in view can read under a harness id, newest first. */
export function lookUpSession(raw: string): Promise<RunSummary[]> {
  const known = lookups.get(raw)
  if (known) return known
  const query = new URLSearchParams({ search: raw, limit: String(LOOKUP_LIMIT), offset: '0' })
  const lookup = fetchInsightsJSON<RunListResponse>(`${RUNS}?${query.toString()}`, 'agent runs')
    .then((list) => sessionMatches(raw, list.data ?? []))
    .then((matches) => {
      // a session that has not reached Replay yet is looked up again next time
      if (matches.length === 0) lookups.delete(raw)
      return matches
    })
    .catch((error: unknown) => {
      lookups.delete(raw)
      throw error
    })
  lookups.set(raw, lookup)
  return lookup
}

export interface SessionScope {
  /** The id the replay API matches: the session's full id, else the value as given. */
  id: string
  /** The value is being matched with the sessions it names. */
  resolving: boolean
  /** The sessions the value names, newest first, when it names more than one. */
  matches: RunSummary[]
  /** Reads another of those sessions instead. */
  choose: (sessionId: string) => void
}

interface Lookup {
  raw: string
  matches?: RunSummary[]
  picked?: string
}

/**
 * The session a link or filter names, by its full id or by the id its harness
 * sent: a harness id reads as the session it names among those the signed-in
 * view can read, the one a link in this tab came from, else the newest.
 */
export function useSessionScope(value: string): SessionScope {
  const raw = value.trim()
  const direct = !raw || isScopedSessionId(raw)
  const known = direct ? undefined : chosenSession(raw)
  const [state, setState] = useState<Lookup>({ raw: '' })

  useEffect(() => {
    if (direct || known) return undefined
    let current = true
    lookUpSession(raw)
      .then((matches) => {
        if (current) setState((previous) => ({ ...previous, raw, matches }))
      })
      .catch(() => {
        if (current) setState((previous) => ({ ...previous, raw, matches: [] }))
      })
    return () => {
      current = false
    }
  }, [direct, known, raw])

  const choose = useCallback(
    (sessionId: string) => {
      sessionParam(sessionId)
      setState((previous) => ({ ...previous, raw, picked: sessionId }))
    },
    [raw],
  )

  if (direct) return { id: raw, resolving: false, matches: [], choose }
  const lookup = state.raw === raw ? state : undefined
  const found = lookup?.matches
  const several = found && found.length > 1 ? found : []
  const picked = lookup?.picked ?? known
  if (picked) return { id: picked, resolving: false, matches: several, choose }
  if (!found) return { id: raw, resolving: true, matches: [], choose }
  return { id: found[0]?.session_id ?? raw, resolving: false, matches: several, choose }
}
