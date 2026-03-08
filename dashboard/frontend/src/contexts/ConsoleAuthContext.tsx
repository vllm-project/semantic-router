import React, { createContext, useContext, useEffect, useState, type ReactNode } from 'react'

interface ConsoleCapabilities {
  canEditConfig: boolean
  canDeployConfig: boolean
  canActivateSetup: boolean
  canRunEvaluation: boolean
  canRunMLPipeline: boolean
  canAdminister: boolean
}

interface ConsoleUser {
  id: string
  email?: string
  displayName?: string
}

interface ConsoleAuthState {
  authenticated: boolean
  authMode: string
  effectiveRole: string
  roles: string[]
  user: ConsoleUser | null
  capabilities: ConsoleCapabilities
  isLoading: boolean
  error: string | null
  refreshSession: () => Promise<void>
}

const emptyCapabilities: ConsoleCapabilities = {
  canEditConfig: false,
  canDeployConfig: false,
  canActivateSetup: false,
  canRunEvaluation: false,
  canRunMLPipeline: false,
  canAdminister: false,
}

const ConsoleAuthContext = createContext<ConsoleAuthState>({
  authenticated: false,
  authMode: '',
  effectiveRole: '',
  roles: [],
  user: null,
  capabilities: emptyCapabilities,
  isLoading: true,
  error: null,
  refreshSession: async () => {},
})

// eslint-disable-next-line react-refresh/only-export-components
export const useConsoleAuth = (): ConsoleAuthState => useContext(ConsoleAuthContext)

interface ConsoleAuthProviderProps {
  children: ReactNode
}

interface AuthSessionResponse {
  authenticated?: boolean
  authMode?: string
  effectiveRole?: string
  roles?: string[]
  user?: ConsoleUser
  capabilities?: Partial<ConsoleCapabilities>
}

async function readErrorMessage(response: Response): Promise<string> {
  const body = await response.text()
  if (!body) {
    return `HTTP ${response.status}: ${response.statusText}`
  }

  try {
    const parsed = JSON.parse(body) as { error?: string; message?: string }
    return parsed.message || parsed.error || body
  } catch {
    return body
  }
}

export const ConsoleAuthProvider: React.FC<ConsoleAuthProviderProps> = ({ children }) => {
  const [authenticated, setAuthenticated] = useState(false)
  const [authMode, setAuthMode] = useState('')
  const [effectiveRole, setEffectiveRole] = useState('')
  const [roles, setRoles] = useState<string[]>([])
  const [user, setUser] = useState<ConsoleUser | null>(null)
  const [capabilities, setCapabilities] = useState<ConsoleCapabilities>(emptyCapabilities)
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const refreshSession = async () => {
    setIsLoading(true)
    setError(null)

    try {
      const response = await fetch('/api/auth/session')
      if (!response.ok) {
        throw new Error(await readErrorMessage(response))
      }

      const session = await response.json() as AuthSessionResponse
      setAuthenticated(Boolean(session.authenticated))
      setAuthMode(session.authMode || '')
      setEffectiveRole(session.effectiveRole || '')
      setRoles(session.roles || [])
      setUser(session.user || null)
      setCapabilities({
        ...emptyCapabilities,
        ...(session.capabilities || {}),
      })
    } catch (err) {
      setAuthenticated(false)
      setAuthMode('')
      setEffectiveRole('')
      setRoles([])
      setUser(null)
      setCapabilities(emptyCapabilities)
      setError(err instanceof Error ? err.message : 'Failed to resolve dashboard session.')
    } finally {
      setIsLoading(false)
    }
  }

  useEffect(() => {
    void refreshSession()
  }, [])

  return (
    <ConsoleAuthContext.Provider
      value={{
        authenticated,
        authMode,
        effectiveRole,
        roles,
        user,
        capabilities,
        isLoading,
        error,
        refreshSession,
      }}
    >
      {children}
    </ConsoleAuthContext.Provider>
  )
}
