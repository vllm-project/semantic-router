import { useEffect, useState } from 'react'

/** Re-renders on an interval so relative times stay fresh. */
export function useNow(intervalMs = 15_000) {
  const [now, setNow] = useState(() => Date.now())
  useEffect(() => {
    const id = window.setInterval(() => setNow(Date.now()), intervalMs)
    return () => window.clearInterval(id)
  }, [intervalMs])
  return now
}

export function timeAgo(iso: string, now = Date.now()): string {
  const time = Date.parse(iso)
  if (!Number.isFinite(time)) return '-'
  const seconds = Math.max(0, Math.round((now - time) / 1000))
  if (seconds < 10) return 'just now'
  if (seconds < 60) return `${seconds}s ago`
  const minutes = Math.round(seconds / 60)
  if (minutes < 60) return `${minutes}m ago`
  const hours = Math.round(minutes / 60)
  if (hours < 48) return `${hours}h ago`
  return `${Math.round(hours / 24)}d ago`
}

/** A recorded duration as people read it: 142ms, 1.72s, 3m 4s. */
export function formatLatency(ms?: number): string {
  if (typeof ms !== 'number' || !Number.isFinite(ms) || ms < 0) return '-'
  if (ms < 1_000) return `${Math.round(ms)}ms`
  if (ms < 60_000) return `${(ms / 1_000).toFixed(ms < 10_000 ? 2 : 1)}s`
  const minutes = Math.floor(ms / 60_000)
  return `${minutes}m ${Math.round((ms % 60_000) / 1_000)}s`
}

export function formatTokenCount(value?: number): string {
  if (typeof value !== 'number' || !Number.isFinite(value)) return '-'
  if (Math.abs(value) < 1000) return String(value)
  return `${(value / 1000).toFixed(value < 10_000 ? 1 : 0)}k`
}

export interface RequestStatusText {
  text: string
  /** The text is an HTTP status code. */
  code: boolean
  /** A success status the request still failed after, for the tooltip. */
  hiddenStatus?: number
}

/**
 * A request's status badge: its HTTP status, unless the request failed after a
 * success status went out (a broken stream, an abort); then its lifecycle.
 */
export function requestStatusText(
  status: number | undefined,
  lifecycle: { successful: boolean; state: string },
): RequestStatusText {
  const code = status ?? 0
  const failedAfterSuccess = !lifecycle.successful && code > 0 && code < 400
  if (code > 0 && !failedAfterSuccess) return { text: String(code), code: true }
  const state = lifecycle.state.replace('_', ' ')
  return {
    text: state.charAt(0).toUpperCase() + state.slice(1),
    code: false,
    hiddenStatus: failedAfterSuccess ? code : undefined,
  }
}
