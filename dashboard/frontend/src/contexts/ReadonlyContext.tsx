import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react'
import { useAuth } from './AuthContext'
import { preloadPlatformAssets } from '../utils/platformAssets'

interface ReadonlyContextType {
  isReadonly: boolean
  serverReadonly: boolean
  runtimeConfigWritable: boolean
  recipeStoreWritable: boolean
  isLoading: boolean
  platform: string
  envoyUrl: string
  routerEvalEndpoint: string
  fleetSimEnabled: boolean
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
  fleetSimEnabled: false,
})

// eslint-disable-next-line react-refresh/only-export-components
export const useReadonly = (): ReadonlyContextType => useContext(ReadonlyContext)

interface ReadonlyProviderProps {
  children: ReactNode
}

export const ReadonlyProvider: React.FC<ReadonlyProviderProps> = ({ children }) => {
  const { token } = useAuth()
  const [isReadonly, setIsReadonly] = useState(true)
  const [serverReadonly, setServerReadonly] = useState(true)
  const [runtimeConfigWritable, setRuntimeConfigWritable] = useState(false)
  const [recipeStoreWritable, setRecipeStoreWritable] = useState(false)
  const [isLoading, setIsLoading] = useState(true)
  const [platform, setPlatform] = useState('')
  const [envoyUrl, setEnvoyUrl] = useState('')
  const [routerEvalEndpoint, setRouterEvalEndpoint] = useState('')
  const [fleetSimEnabled, setFleetSimEnabled] = useState(false)

  useEffect(() => {
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
      try {
        const response = await fetch('/api/settings', { signal: controller.signal })
        if (response.ok) {
          const data = await response.json()
          if (controller.signal.aborted) return
          if (typeof data.readonlyMode !== 'boolean') {
            console.warn('Dashboard settings omitted the readonlyMode safety boundary')
            return
          }
          const effectiveReadonly = data.readonlyMode
          // Older Dashboard responses exposed only readonlyMode. Preserve the
          // safe all-or-nothing fallback while consuming split capabilities
          // whenever the server provides them.
          setIsReadonly(effectiveReadonly)
          setServerReadonly(
            typeof data.serverReadonly === 'boolean' ? data.serverReadonly : effectiveReadonly,
          )
          setRuntimeConfigWritable(
            typeof data.runtimeConfigWritable === 'boolean'
              ? data.runtimeConfigWritable
              : !effectiveReadonly,
          )
          setRecipeStoreWritable(
            typeof data.recipeStoreWritable === 'boolean'
              ? data.recipeStoreWritable
              : !effectiveReadonly,
          )
          const platformValue = data.platform || ''
          setPlatform(platformValue)
          setEnvoyUrl(data.envoyUrl || '')
          setRouterEvalEndpoint(data.routerEvalEndpoint || '')
          setFleetSimEnabled(Boolean(data.fleetSimEnabled))
          // Preload platform-specific assets immediately
          preloadPlatformAssets(platformValue)
        }
      } catch (error) {
        if (!controller.signal.aborted) {
          console.warn('Failed to fetch dashboard settings:', error)
        }
      } finally {
        if (!controller.signal.aborted) setIsLoading(false)
      }
    }

    void fetchSettings()
    return () => controller.abort()
  }, [token])

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
        fleetSimEnabled,
      }}
    >
      {children}
    </ReadonlyContext.Provider>
  )
}
