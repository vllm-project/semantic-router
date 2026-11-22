import {
  insightsUnavailableReason,
  InsightsRequestError,
  type InsightsUnavailableReason,
} from './insightsPageApi'
import type { RunResponse } from './insightsRunTypes'

/** What the run page shows, and the read that said it. */
export interface RunReadState {
  /** The session this state describes; a read for another session leaves it alone. */
  sessionId: string
  /** The newest read started; a read settling under an older number no longer applies. */
  read: number
  response: RunResponse | null
  error: string | null
  unavailable?: InsightsUnavailableReason
  notFound: boolean
  loading: boolean
}

/** One read of one session, numbered as the page started it. */
export interface RunRead {
  sessionId: string
  number: number
}

export type RunReadOutcome = { response: RunResponse } | { error: unknown }

/** A page that names this session and has not been answered by a read yet. */
export function initialRunReads(sessionId: string): RunReadState {
  return { sessionId, read: 0, response: null, error: null, notFound: false, loading: true }
}

/** Starts one read: a refresh keeps its run on screen, another session starts clean. */
export function beginRunRead(previous: RunReadState, read: RunRead): RunReadState {
  const state = previous.sessionId === read.sessionId ? previous : initialRunReads(read.sessionId)
  return { ...state, read: read.number, loading: true }
}

/** Applies one read's answer only while no newer read has started since. */
export function settleRunRead(
  state: RunReadState,
  read: RunRead,
  outcome: RunReadOutcome,
): RunReadState {
  if (state.sessionId !== read.sessionId || state.read !== read.number) return state
  if ('response' in outcome) {
    return {
      ...state,
      response: outcome.response,
      error: null,
      notFound: false,
      unavailable: undefined,
      loading: false,
    }
  }
  const error = outcome.error
  const reason = insightsUnavailableReason(error)
  const notFound = error instanceof InsightsRequestError && error.status === 404
  return {
    ...state,
    // a failed refresh keeps the run on screen under its error
    response: notFound ? null : state.response,
    error: reason ? null : error instanceof Error ? error.message : 'Unknown error',
    unavailable: notFound ? undefined : reason,
    notFound,
    loading: false,
  }
}

/** The state a page naming this session shows: another session's read stays off it. */
export function shownRunReads(state: RunReadState, sessionId: string): RunReadState {
  return state.sessionId === sessionId ? state : initialRunReads(sessionId)
}
