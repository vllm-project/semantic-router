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
