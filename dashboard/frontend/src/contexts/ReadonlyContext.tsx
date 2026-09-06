import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react'
import { useAuth } from './AuthContext'
import { preloadPlatformAssets } from '../utils/platformAssets'
import { decodeDashboardSettings } from './dashboardSettings'

interface ReadonlyContextType {
  isReadonly: boolean
  serverReadonly: boolean
  runtimeConfigWritable: boolean
  recipeStoreWritable: boolean
  isLoading: boolean
  platform: string
  envoyUrl: string
  routerEvalEndpoint: string
  evaluationAvailable: boolean
  evaluationUnavailableReason: string
}

const ReadonlyContext = createContext<ReadonlyContextType>({
  isReadonly: true,
  serverReadonly: true,
  runtimeConfigWritable: false,
  recipeStoreWritable: false,
  isLoading: true,
  platform: '',
  envoyUrl: '',
  routerEvalEndpoint: '',
  evaluationAvailable: false,
  evaluationUnavailableReason: 'Evaluation availability has not been loaded.',
})

// eslint-disable-next-line react-refresh/only-export-components
export const useReadonly = (): ReadonlyContextType => useContext(ReadonlyContext)

interface ReadonlyProviderProps {
  children: ReactNode
}

export const ReadonlyProvider: React.FC<ReadonlyProviderProps> = ({ children }) => {
  const { isAuthenticated } = useAuth()
  const [isReadonly, setIsReadonly] = useState(true)
  const [serverReadonly, setServerReadonly] = useState(true)
  const [runtimeConfigWritable, setRuntimeConfigWritable] = useState(false)
  const [recipeStoreWritable, setRecipeStoreWritable] = useState(false)
  const [isLoading, setIsLoading] = useState(true)
  const [platform, setPlatform] = useState('')
  const [envoyUrl, setEnvoyUrl] = useState('')
  const [routerEvalEndpoint, setRouterEvalEndpoint] = useState('')
  const [evaluationAvailable, setEvaluationAvailable] = useState(false)
  const [evaluationUnavailableReason, setEvaluationUnavailableReason] = useState(
    'Evaluation availability has not been loaded.',
  )

  useEffect(() => {
    if (!isAuthenticated) {
      setIsReadonly(true)
      setServerReadonly(true)
      setRuntimeConfigWritable(false)
      setRecipeStoreWritable(false)
      setEvaluationAvailable(false)
      setEvaluationUnavailableReason('Evaluation is unavailable without an authenticated session.')
      setPlatform('')
      setEnvoyUrl('')
      setRouterEvalEndpoint('')
      setIsLoading(false)
      return undefined
    }

    const controller = new AbortController()
    const fetchSettings = async () => {
      setIsLoading(true)
      // Settings are part of the mutation authorization boundary. Never keep
      // capabilities from a previous session while a refresh is pending, and
      // keep every mutation surface closed if the request fails.
      setIsReadonly(true)
      setServerReadonly(true)
      setRuntimeConfigWritable(false)
      setRecipeStoreWritable(false)
      setEvaluationAvailable(false)
      setEvaluationUnavailableReason('Evaluation availability is being checked.')
      try {
        const response = await fetch('/api/settings', { signal: controller.signal })
        if (!response.ok) throw new Error(`Dashboard settings request failed (${response.status})`)
        const data = decodeDashboardSettings(await response.json())
        if (controller.signal.aborted) return
        setIsReadonly(data.readonlyMode)
        setServerReadonly(data.serverReadonly)
        setRuntimeConfigWritable(data.runtimeConfigWritable)
        setRecipeStoreWritable(data.recipeStoreWritable)
        setEvaluationAvailable(data.evaluationAvailable)
        setEvaluationUnavailableReason(data.evaluationUnavailableReason)
        const platformValue = data.platform
        setPlatform(platformValue)
        setEnvoyUrl(data.envoyUrl)
        setRouterEvalEndpoint(data.routerEvalEndpoint)
        preloadPlatformAssets(platformValue)
      } catch (error) {
        if (!controller.signal.aborted) {
          setEvaluationUnavailableReason('Dashboard settings are unavailable.')
          console.warn('Failed to fetch dashboard settings:', error)
        }
      } finally {
        if (!controller.signal.aborted) setIsLoading(false)
      }
    }

    void fetchSettings()
    return () => controller.abort()
  }, [isAuthenticated])

  return (
    <ReadonlyContext.Provider
      value={{
        isReadonly,
        serverReadonly,
        runtimeConfigWritable,
        recipeStoreWritable,
        isLoading,
        platform,
        envoyUrl,
        routerEvalEndpoint,
        evaluationAvailable,
        evaluationUnavailableReason,
      }}
    >
      {children}
    </ReadonlyContext.Provider>
  )
}
