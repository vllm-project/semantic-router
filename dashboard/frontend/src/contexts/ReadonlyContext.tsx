import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react'

interface ReadonlyContextType {
  isReadonly: boolean
  isLoading: boolean
}

const ReadonlyContext = createContext<ReadonlyContextType>({
  isReadonly: false,
  isLoading: true,
})

export const useReadonly = (): ReadonlyContextType => useContext(ReadonlyContext)

interface ReadonlyProviderProps {
  children: ReactNode
}

export const ReadonlyProvider: React.FC<ReadonlyProviderProps> = ({ children }) => {
  const [isReadonly, setIsReadonly] = useState(false)
  const [isLoading, setIsLoading] = useState(true)

  useEffect(() => {
    const fetchSettings = async () => {
      try {
        const response = await fetch('/api/settings')
        if (response.ok) {
          const data = await response.json()
          setIsReadonly(data.readonlyMode || false)
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
    <ReadonlyContext.Provider value={{ isReadonly, isLoading }}>
      {children}
    </ReadonlyContext.Provider>
  )
}
