import { useEffect, useState } from 'react'
import type { SystemStatus } from '../utils/routerRuntime'
import type { StatusTone } from '../ui'

export interface ShellStatus {
  status: SystemStatus | null
  tone: StatusTone
  label: string
  reachable: boolean
}

const POLL_MS = 15_000

function describe(
  status: SystemStatus | null,
  failed: boolean,
): Pick<ShellStatus, 'tone' | 'label'> {
  if (failed) return { tone: 'danger', label: 'Dashboard offline' }
  if (!status) return { tone: 'neutral', label: 'Checking…' }
  const overall = status.overall?.toLowerCase()
  if (overall === 'healthy') return { tone: 'success', label: 'Router healthy' }
  if (overall === 'degraded') return { tone: 'warning', label: 'Degraded' }
  if (overall === 'starting') return { tone: 'info', label: 'Starting' }
  if (overall === 'standby') return { tone: 'neutral', label: 'Standby' }
  return { tone: 'danger', label: 'Router not running' }
}

/** Polls `/api/status` for the shell's instance card and status pill; pauses while the tab is hidden. */
export function useShellStatus(): ShellStatus {
  const [status, setStatus] = useState<SystemStatus | null>(null)
  const [failed, setFailed] = useState(false)

  useEffect(() => {
    let cancelled = false
    let controller: AbortController | null = null
    const load = async () => {
      if (document.hidden) return
      controller?.abort()
      controller = new AbortController()
      try {
        const response = await fetch('/api/status', { signal: controller.signal })
        if (!response.ok) throw new Error(String(response.status))
        const data = (await response.json()) as SystemStatus
        if (!cancelled) {
          setStatus(data)
          setFailed(false)
        }
      } catch (error) {
        if (!cancelled && !(error instanceof DOMException && error.name === 'AbortError'))
          setFailed(true)
      }
    }
    void load()
    const interval = window.setInterval(() => void load(), POLL_MS)
    const onVisible = () => {
      if (!document.hidden) void load()
    }
    document.addEventListener('visibilitychange', onVisible)
    return () => {
      cancelled = true
      controller?.abort()
      window.clearInterval(interval)
      document.removeEventListener('visibilitychange', onVisible)
    }
  }, [])

  return { status, reachable: !failed, ...describe(status, failed) }
}
