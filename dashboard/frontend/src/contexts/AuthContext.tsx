import React, {
  createContext,
  ReactNode,
  useCallback,
  useContext,
  useEffect,
  useState,
} from 'react'
import {
  clearStoredAuthToken,
  getAuthSessionRevision,
  getStoredAuthToken,
  installAuthenticatedFetch,
  normalizeAuthToken,
  notifyUnauthorized,
  shouldClearSessionForUnauthorized,
  storeAuthToken,
  UNAUTHORIZED_EVENT,
} from '../utils/authFetch'
import {
  changePasswordAndRotateSession,
  fetchCurrentAuthUser,
  hasAuthenticatedSession,
  readAuthResponseError,
  type AuthUser,
} from './authSession'

interface AuthContextValue {
  token: string | null
  user: AuthUser | null
  isLoading: boolean
  isAuthenticated: boolean
  login: (email: string, password: string) => Promise<void>
  changePassword: (currentPassword: string, newPassword: string) => Promise<void>
  setSession: (token: string, user?: AuthUser | null) => void
  logout: () => void
  refreshSession: () => Promise<void>
}

const AuthContext = createContext<AuthContextValue | undefined>(undefined)

installAuthenticatedFetch()

export const AuthProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  const [token, setToken] = useState<string | null>(() => getStoredAuthToken())
  const [user, setUser] = useState<AuthUser | null>(null)
  const [isLoading, setIsLoading] = useState(true)

  const clearSession = useCallback(() => {
    setToken(null)
    setUser(null)
    clearStoredAuthToken()
  }, [])

  const setSession = useCallback((nextToken: string, nextUser?: AuthUser | null) => {
    const storedToken = storeAuthToken(nextToken)
    setToken(storedToken)
    setUser(storedToken ? (nextUser ?? null) : null)
  }, [])

  const changePassword = useCallback(
    (currentPassword: string, newPassword: string) =>
      changePasswordAndRotateSession(currentPassword, newPassword, setSession, user),
    [setSession, user],
  )

  const refreshSession = useCallback(async () => {
    const requestRevision = getAuthSessionRevision()
    setIsLoading(true)
    try {
      const result = await fetchCurrentAuthUser()
      if (result.clearLocalToken) {
        notifyUnauthorized(requestRevision)
        return
      }
      setUser(result.user)
    } catch {
      notifyUnauthorized(requestRevision)
    } finally {
      setIsLoading(false)
    }
  }, [])

  useEffect(() => {
    if (token) {
      const storedToken = storeAuthToken(token)
      if (storedToken !== token) {
        setToken(storedToken)
      }
    }
  }, [token])

  useEffect(() => {
    void refreshSession()
  }, [refreshSession])

  useEffect(() => {
    const handleUnauthorized = (event: Event) => {
      if (!shouldClearSessionForUnauthorized(getAuthSessionRevision(), event)) {
        return
      }
      clearSession()
      setIsLoading(false)
    }

    window.addEventListener(UNAUTHORIZED_EVENT, handleUnauthorized)
    return () => window.removeEventListener(UNAUTHORIZED_EVENT, handleUnauthorized)
  }, [clearSession])

  const login = async (email: string, password: string) => {
    setIsLoading(true)
    try {
      const response = await fetch('/api/auth/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, password }),
      })

      if (!response.ok) {
        throw new Error(await readAuthResponseError(response))
      }

      const payload = (await response.json()) as { token: string; user?: AuthUser }
      const nextToken = normalizeAuthToken(payload.token)
      if (!nextToken) {
        throw new Error('Login response did not include a valid session token')
      }
      setSession(nextToken, payload.user ?? null)
    } finally {
      setIsLoading(false)
    }
  }

  const logout = () => {
    void fetch('/api/auth/logout', { method: 'POST', keepalive: true }).catch(() => {
      // Local logout should still complete if the server session clear cannot be reached.
    })
    clearSession()
  }

  return (
    <AuthContext.Provider
      value={{
        token,
        user,
        isLoading,
        isAuthenticated: hasAuthenticatedSession(token, user),
        login,
        changePassword,
        setSession,
        logout,
        refreshSession,
      }}
    >
      {children}
    </AuthContext.Provider>
  )
}

export const useAuth = (): AuthContextValue => {
  const value = useContext(AuthContext)
  if (!value) {
    throw new Error('useAuth must be used within an AuthProvider')
  }
  return value
}
