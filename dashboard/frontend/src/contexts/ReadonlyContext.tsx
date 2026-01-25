import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react'
import { preloadPlatformAssets } from '../components/PlatformBranding'

interface ReadonlyContextType {
  isReadonly: boolean
  isLoading: boolean
  platform: string
}

const ReadonlyContext = createContext<ReadonlyContextType>({
  isReadonly: false,
  isLoading: true,
  platform: '',
})

export const useReadonly = (): ReadonlyContextType => useContext(ReadonlyContext)

interface ReadonlyProviderProps {
  children: ReactNode
}

export const ReadonlyProvider: React.FC<ReadonlyProviderProps> = ({ children }) => {
  const [isReadonly, setIsReadonly] = useState(false)
  const [isLoading, setIsLoading] = useState(true)
  const [platform, setPlatform] = useState('')

  useEffect(() => {
    const fetchSettings = async () => {
      try {
        const response = await fetch('/api/settings')
        if (response.ok) {
          const data = await response.json()
          setIsReadonly(data.readonlyMode || false)
          const platformValue = data.platform || ''
          setPlatform(platformValue)
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
  }, [])

  return (
    <ReadonlyContext.Provider value={{ isReadonly, isLoading, platform }}>
      {children}
    </ReadonlyContext.Provider>
  )
}
