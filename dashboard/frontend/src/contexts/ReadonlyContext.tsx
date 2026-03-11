import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react'
import { useAuth } from './AuthContext'
import { preloadPlatformAssets } from '../utils/platformAssets'

interface ReadonlyContextType {
  isReadonly: boolean
  isLoading: boolean
  platform: string
  envoyUrl: string
}

const ReadonlyContext = createContext<ReadonlyContextType>({
  isReadonly: false,
  isLoading: true,
  platform: '',
  envoyUrl: '',
})

// eslint-disable-next-line react-refresh/only-export-components
export const useReadonly = (): ReadonlyContextType => useContext(ReadonlyContext)

interface ReadonlyProviderProps {
  children: ReactNode
}

export const ReadonlyProvider: React.FC<ReadonlyProviderProps> = ({ children }) => {
  const { token } = useAuth()
  const [isReadonly, setIsReadonly] = useState(false)
  const [isLoading, setIsLoading] = useState(true)
  const [platform, setPlatform] = useState('')
  const [envoyUrl, setEnvoyUrl] = useState('')

  useEffect(() => {
    const fetchSettings = async () => {
      setIsLoading(true)
      try {
        const response = await fetch('/api/settings')
        if (response.ok) {
          const data = await response.json()
          setIsReadonly(data.readonlyMode || false)
          const platformValue = data.platform || ''
          setPlatform(platformValue)
          setEnvoyUrl(data.envoyUrl || '')
          // Preload platform-specific assets immediately
          preloadPlatformAssets(platformValue)
        }
      } catch (error) {
        console.warn('Failed to fetch dashboard settings:', error)
      } finally {
        setIsLoading(false)
      }
    }

    fetchSettings()
  }, [token])

  return (
    <ReadonlyContext.Provider value={{ isReadonly, isLoading, platform, envoyUrl }}>
      {children}
    </ReadonlyContext.Provider>
  )
}
