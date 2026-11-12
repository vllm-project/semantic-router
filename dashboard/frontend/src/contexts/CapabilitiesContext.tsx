import React, {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useState,
  type ReactNode,
} from 'react'
import { useAuth } from './AuthContext'
import { withRequestTimeout } from '../utils/boundedRequest'
import {
  decodeDashboardCapabilities,
  PROBE_PENDING,
  type CapabilityId,
  type DashboardCapabilities,
  type DashboardCapability,
} from './dashboardCapabilities'

// The server publishes its first health checks within seconds of starting,
// so a pending probe is re-read briefly instead of leaving a feature closed.
const PROBE_PENDING_RETRY_MS = 2_000
const PROBE_PENDING_MAX_RETRIES = 5

interface CapabilitiesContextType {
  capabilities: DashboardCapabilities | null
  isLoading: boolean
  error: string | null
  refreshCapabilities: () => void
}

const CapabilitiesContext = createContext<CapabilitiesContextType>({
  capabilities: null,
  isLoading: true,
  error: null,
  refreshCapabilities: () => {},
})

// eslint-disable-next-line react-refresh/only-export-components
export const useCapabilities = (): CapabilitiesContextType => useContext(CapabilitiesContext)

// eslint-disable-next-line react-refresh/only-export-components
export const useCapability = (id: CapabilityId): DashboardCapability | undefined =>
  useContext(CapabilitiesContext).capabilities?.[id]

export const CapabilitiesProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  const { isAuthenticated, user } = useAuth()
  const userID = user?.id
  const accessSnapshot = JSON.stringify([user?.role, user?.permissions])
  const [capabilities, setCapabilities] = useState<DashboardCapabilities | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [revision, setRevision] = useState(0)
  const [pendingRetries, setPendingRetries] = useState(0)

  const refreshCapabilities = useCallback(() => {
    setPendingRetries(0)
    setIsLoading(true)
    setRevision((current) => current + 1)
  }, [])

  // Never keep a previous session's capabilities: clear them when the caller
  // or their access changes, before the next read completes.
  useEffect(() => {
    setCapabilities(null)
    setError(null)
    setPendingRetries(0)
    setIsLoading(isAuthenticated)
  }, [isAuthenticated, userID, accessSnapshot])

  useEffect(() => {
    if (!isAuthenticated) return undefined
    const controller = new AbortController()
    let retry: ReturnType<typeof setTimeout> | undefined
    let failureMessage = 'Dashboard feature availability is unavailable. Refresh access to retry.'
    void (async () => {
      try {
        const decoded = await withRequestTimeout(async (signal) => {
          const response = await fetch('/api/v1/capabilities', { signal })
          if (!response.ok) {
            if (response.status === 403) {
              failureMessage =
                'Access to Dashboard feature availability was denied. Refresh access or contact an administrator.'
            }
            throw new Error(`Capabilities request failed (${response.status})`)
          }
          return decodeDashboardCapabilities(await response.json())
        }, controller.signal)
        if (controller.signal.aborted) return
        setCapabilities(decoded)
        setError(null)
        const pending = Object.values(decoded).some((capability) => capability.reason === PROBE_PENDING)
        if (pending && pendingRetries < PROBE_PENDING_MAX_RETRIES) {
          retry = setTimeout(() => {
            setPendingRetries((current) => current + 1)
            setRevision((current) => current + 1)
          }, PROBE_PENDING_RETRY_MS)
        }
      } catch (err) {
        if (controller.signal.aborted) return
        setCapabilities(null)
        setError(failureMessage)
        console.warn('Failed to fetch dashboard capabilities:', err)
      } finally {
        if (!controller.signal.aborted) setIsLoading(false)
      }
    })()
    return () => {
      controller.abort()
      if (retry) clearTimeout(retry)
    }
    // pendingRetries only bounds the retries this effect schedules.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isAuthenticated, userID, accessSnapshot, revision])

  return (
    <CapabilitiesContext.Provider value={{ capabilities, isLoading, error, refreshCapabilities }}>
      {children}
    </CapabilitiesContext.Provider>
  )
}
