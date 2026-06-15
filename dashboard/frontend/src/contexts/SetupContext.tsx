import React, { createContext, ReactNode, useCallback, useContext, useEffect, useState } from 'react'
import { SetupState } from '../types/setup'
import { fetchSetupState } from '../utils/setupApi'

interface SetupContextValue {
  setupState: SetupState | null
  isLoading: boolean
  error: string | null
  refreshSetupState: () => Promise<SetupState | null>
}

const SetupContext = createContext<SetupContextValue>({
  setupState: null,
  isLoading: true,
  error: null,
  refreshSetupState: async () => null,
})

// eslint-disable-next-line react-refresh/only-export-components
export const useSetup = (): SetupContextValue => useContext(SetupContext)

interface SetupProviderProps {
  children: ReactNode
}

export const SetupProvider: React.FC<SetupProviderProps> = ({ children }) => {
  const [setupState, setSetupState] = useState<SetupState | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const refreshSetupState = useCallback(async (): Promise<SetupState | null> => {
    try {
      setError(null)
      const nextState = await fetchSetupState()
      setSetupState(nextState)
      return nextState
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to fetch setup state'
      setError(message)
      setSetupState(null)
      return null
    } finally {
      setIsLoading(false)
    }
  }, [])

  useEffect(() => {
    void refreshSetupState()
  }, [refreshSetupState])

  return (
    <SetupContext.Provider value={{ setupState, isLoading, error, refreshSetupState }}>
      {children}
    </SetupContext.Provider>
  )
}
