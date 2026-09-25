import React, {
  createContext,
  useCallback,
  useContext,
  useState,
  useEffect,
  ReactNode,
} from 'react'
import { useAuth } from './AuthContext'
import { preloadPlatformAssets } from '../utils/platformAssets'
import { decodeDashboardSettings } from './dashboardSettings'

interface ReadonlyContextType {
  isReadonly: boolean
  serverReadonly: boolean
  runtimeConfigWritable: boolean
  recipeStoreWritable: boolean
  isLoading: boolean
  settingsError: string | null
  refreshSettings: () => void
  platform: string
  envoyUrl: string
  routerEvalEndpoint: string
  srBenchAvailable: boolean
  srBenchUnavailableReason: string
}

const ReadonlyContext = createContext<ReadonlyContextType>({
  isReadonly: true,
  serverReadonly: true,
  runtimeConfigWritable: false,
  recipeStoreWritable: false,
  isLoading: true,
  settingsError: null,
  refreshSettings: () => {},
  platform: '',
  envoyUrl: '',
  routerEvalEndpoint: '',
  srBenchAvailable: false,
  srBenchUnavailableReason: 'Evaluation availability has not been loaded.',
})

// eslint-disable-next-line react-refresh/only-export-components
export const useReadonly = (): ReadonlyContextType => useContext(ReadonlyContext)

interface ReadonlyProviderProps {
  children: ReactNode
}

export const ReadonlyProvider: React.FC<ReadonlyProviderProps> = ({ children }) => {
  const { isAuthenticated, user } = useAuth()
  const userID = user?.id
  const accessSnapshot = JSON.stringify([user?.role, user?.permissions])
  const [isReadonly, setIsReadonly] = useState(true)
  const [serverReadonly, setServerReadonly] = useState(true)
  const [runtimeConfigWritable, setRuntimeConfigWritable] = useState(false)
  const [recipeStoreWritable, setRecipeStoreWritable] = useState(false)
  const [isLoading, setIsLoading] = useState(true)
  const [settingsError, setSettingsError] = useState<string | null>(null)
  const [settingsRevision, setSettingsRevision] = useState(0)
  const [platform, setPlatform] = useState('')
  const [envoyUrl, setEnvoyUrl] = useState('')
  const [routerEvalEndpoint, setRouterEvalEndpoint] = useState('')
  const [srBenchAvailable, setSrBenchAvailable] = useState(false)
  const [srBenchUnavailableReason, setSrBenchUnavailableReason] = useState(
    'Evaluation availability has not been loaded.',
  )

  const refreshSettings = useCallback(() => {
    setIsReadonly(true)
    setServerReadonly(true)
    setRuntimeConfigWritable(false)
    setRecipeStoreWritable(false)
    setSrBenchAvailable(false)
    setSettingsError(null)
    setIsLoading(true)
    setSettingsRevision((revision) => revision + 1)
  }, [])

  useEffect(() => {
    if (!isAuthenticated) {
      setIsReadonly(true)
      setServerReadonly(true)
      setRuntimeConfigWritable(false)
      setRecipeStoreWritable(false)
      setSrBenchAvailable(false)
      setSrBenchUnavailableReason('Evaluation is unavailable without an authenticated session.')
      setPlatform('')
      setEnvoyUrl('')
      setRouterEvalEndpoint('')
      setSettingsError(null)
      setIsLoading(false)
      return undefined
    }

    const controller = new AbortController()
    const fetchSettings = async () => {
      setIsLoading(true)
      setSettingsError(null)
      // Settings are part of the mutation authorization boundary. Never keep
      // capabilities from a previous session while a refresh is pending, and
      // keep every mutation surface closed if the request fails.
      setIsReadonly(true)
      setServerReadonly(true)
      setRuntimeConfigWritable(false)
      setRecipeStoreWritable(false)
      setSrBenchAvailable(false)
      setSrBenchUnavailableReason('Evaluation availability is being checked.')
      let failureMessage = 'Dashboard access settings are unavailable. Refresh access to retry.'
      try {
        const response = await fetch('/api/settings', { signal: controller.signal })
        if (!response.ok) {
          if (response.status === 403) {
            failureMessage =
              'Access to Dashboard settings was denied. Refresh access or contact an administrator.'
          }
          throw new Error(`Dashboard settings request failed (${response.status})`)
        }
        const data = decodeDashboardSettings(await response.json())
        if (controller.signal.aborted) return
        setSettingsError(null)
        setIsReadonly(data.readonlyMode)
        setServerReadonly(data.serverReadonly)
        setRuntimeConfigWritable(data.runtimeConfigWritable)
        setRecipeStoreWritable(data.recipeStoreWritable)
        setSrBenchAvailable(data.srBenchAvailable)
        setSrBenchUnavailableReason(data.srBenchUnavailableReason)
        const platformValue = data.platform
        setPlatform(platformValue)
        setEnvoyUrl(data.envoyUrl)
        setRouterEvalEndpoint(data.routerEvalEndpoint)
        preloadPlatformAssets(platformValue)
      } catch (error) {
        if (!controller.signal.aborted) {
          setSettingsError(failureMessage)
          setSrBenchUnavailableReason('Dashboard settings are unavailable.')
          console.warn('Failed to fetch dashboard settings:', error)
        }
      } finally {
        if (!controller.signal.aborted) setIsLoading(false)
      }
    }

    void fetchSettings()
    return () => controller.abort()
  }, [isAuthenticated, userID, accessSnapshot, settingsRevision])

  return (
    <ReadonlyContext.Provider
      value={{
        isReadonly,
        serverReadonly,
        runtimeConfigWritable,
        recipeStoreWritable,
        isLoading,
        settingsError,
        refreshSettings,
        platform,
        envoyUrl,
        routerEvalEndpoint,
        srBenchAvailable,
        srBenchUnavailableReason,
      }}
    >
      {children}
    </ReadonlyContext.Provider>
  )
}
