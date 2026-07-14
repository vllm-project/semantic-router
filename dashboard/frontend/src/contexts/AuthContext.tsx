import React, {
  createContext,
  ReactNode,
  useCallback,
  useContext,
  useEffect,
  useState,
} from 'react'
import {
  getAuthSessionRevision,
  installAuthenticatedFetch,
  markAuthSessionChanged,
  notifyUnauthorized,
  shouldClearSessionForUnauthorized,
  UNAUTHORIZED_EVENT,
} from '../utils/authFetch'
import {
  changePasswordAndRotateSession,
  COOKIE_AUTH_RESPONSE_HEADERS,
  fetchCurrentAuthUser,
  hasAuthenticatedSession,
  readAuthResponseError,
  type AuthUser,
} from './authSession'

interface AuthContextValue {
  user: AuthUser | null
  isLoading: boolean
  isAuthenticated: boolean
  login: (email: string, password: string) => Promise<void>
  changePassword: (currentPassword: string, newPassword: string) => Promise<void>
  establishSession: (user?: AuthUser | null) => Promise<void>
  logout: () => Promise<void>
  refreshSession: () => Promise<void>
}

const AuthContext = createContext<AuthContextValue | undefined>(undefined)

installAuthenticatedFetch()

export const AuthProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  const [user, setUser] = useState<AuthUser | null>(null)
  const [isLoading, setIsLoading] = useState(true)

  const clearSession = useCallback(() => {
    setUser(null)
    markAuthSessionChanged()
  }, [])

  const establishSession = useCallback(async (nextUser?: AuthUser | null) => {
    // Advance before any fallback request so a late 401 from the previous
    // session cannot clear the newly issued HttpOnly cookie.
    markAuthSessionChanged()
    if (nextUser) {
      setUser(nextUser)
      return
    }

    const result = await fetchCurrentAuthUser()
    if (!result.user) {
      throw new Error('Authentication succeeded but the session user is unavailable')
    }
    setUser(result.user)
  }, [])

  const changePassword = useCallback(
    (currentPassword: string, newPassword: string) =>
      changePasswordAndRotateSession(currentPassword, newPassword, establishSession, user),
    [establishSession, user],
  )

  const refreshSession = useCallback(async () => {
    const requestRevision = getAuthSessionRevision()
    setIsLoading(true)
    try {
      const result = await fetchCurrentAuthUser()
      if (result.unauthorized) {
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
        credentials: 'same-origin',
        cache: 'no-store',
        redirect: 'error',
        headers: {
          'Content-Type': 'application/json',
          ...COOKIE_AUTH_RESPONSE_HEADERS,
        },
        body: JSON.stringify({ email, password }),
      })

      if (!response.ok) {
        throw new Error(await readAuthResponseError(response))
      }

      const payload = (await response.json()) as { user?: AuthUser }
      await establishSession(payload.user ?? null)
    } finally {
      setIsLoading(false)
    }
  }

  const logout = async () => {
    const response = await fetch('/api/auth/logout', {
      method: 'POST',
      credentials: 'same-origin',
      cache: 'no-store',
      keepalive: true,
    })
    if (!response.ok) {
      throw new Error(await readAuthResponseError(response))
    }
    clearSession()
  }

  return (
    <AuthContext.Provider
      value={{
        user,
        isLoading,
        isAuthenticated: hasAuthenticatedSession(user),
        login,
        changePassword,
        establishSession,
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
