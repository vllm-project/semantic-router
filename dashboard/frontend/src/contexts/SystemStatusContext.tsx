import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from 'react'

import {
  fetchSystemStatus,
  isRoutingAccessOperational,
  type SystemStatus,
} from '../utils/routerRuntime'

export type RoutingAccessAvailability = 'checking' | 'operational' | 'unavailable'

interface SystemStatusContextValue {
  status: SystemStatus | null
  isLoading: boolean
  error: string | null
  lastUpdated: Date | null
  routingAccess: RoutingAccessAvailability
  refresh: () => Promise<void>
}

const SystemStatusContext = createContext<SystemStatusContextValue | undefined>(undefined)
const STATUS_REFRESH_INTERVAL_MS = 10_000

export function resolveRoutingAccessAvailability(
  status: SystemStatus | null,
  error: string | null,
  isLoading: boolean,
): RoutingAccessAvailability {
  if (error) return 'unavailable'
  if (!status) return isLoading ? 'checking' : 'unavailable'
  return isRoutingAccessOperational(status) ? 'operational' : 'unavailable'
}

export function SystemStatusProvider({ children }: { children: ReactNode }) {
  const [status, setStatus] = useState<SystemStatus | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null)
  const requestGeneration = useRef(0)
  const mounted = useRef(true)

  const refresh = useCallback(async () => {
    const generation = ++requestGeneration.current
    try {
      const nextStatus = await fetchSystemStatus()
      if (!mounted.current || generation !== requestGeneration.current) return
      setStatus(nextStatus)
      setLastUpdated(new Date())
      setError(null)
    } catch (cause) {
      if (!mounted.current || generation !== requestGeneration.current) return
      setError(cause instanceof Error ? cause.message : 'System status is unavailable.')
    } finally {
      if (mounted.current && generation === requestGeneration.current) setIsLoading(false)
    }
  }, [])

  useEffect(() => {
    mounted.current = true
    void refresh()

    const refreshWhenVisible = () => {
      if (!document.hidden) void refresh()
    }
    document.addEventListener('visibilitychange', refreshWhenVisible)
    const interval = window.setInterval(refreshWhenVisible, STATUS_REFRESH_INTERVAL_MS)

    return () => {
      mounted.current = false
      requestGeneration.current += 1
      window.clearInterval(interval)
      document.removeEventListener('visibilitychange', refreshWhenVisible)
    }
  }, [refresh])

  const routingAccess = resolveRoutingAccessAvailability(status, error, isLoading)
  const value = useMemo<SystemStatusContextValue>(
    () => ({ status, isLoading, error, lastUpdated, routingAccess, refresh }),
    [error, isLoading, lastUpdated, refresh, routingAccess, status],
  )

  return <SystemStatusContext.Provider value={value}>{children}</SystemStatusContext.Provider>
}

export function useSystemStatus(): SystemStatusContextValue {
  const value = useContext(SystemStatusContext)
  if (!value) throw new Error('useSystemStatus must be used within SystemStatusProvider')
  return value
}
