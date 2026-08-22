import React, { createContext, ReactNode, useCallback, useContext, useEffect, useState } from 'react'
import {
  installAuthenticatedFetch,
  normalizeAuthToken,
  notifyUnauthorized,
  UNAUTHORIZED_EVENT,
} from '../utils/authFetch'
import {
  fetchCurrentAuthUser,
  hasAuthenticatedSession,
  type AuthUser,
} from './authSession'

interface AuthContextValue {
  token: string | null
  user: AuthUser | null
  isLoading: boolean
  isAuthenticated: boolean
  login: (email: string, password: string) => Promise<void>
  setSession: (token: string, user?: AuthUser | null) => void
  logout: () => void
  refreshSession: () => Promise<void>
}

const AuthContext = createContext<AuthContextValue | undefined>(undefined)

const readErrorMessage = async (response: Response): Promise<string> => {
  const body = await response.text()
  if (!body) {
    return `HTTP ${response.status}: ${response.statusText}`
  }

  try {
    const payload = JSON.parse(body) as { message?: string; error?: string }
    return payload.message ?? payload.error ?? body
  } catch {
    return body
  }
}

installAuthenticatedFetch()

export const AuthProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  // The session lives in the HttpOnly cookie, which we cannot read; /api/auth/me answers
  // whether we are logged in. `token` is state only for the moment between a successful
  // login response and the refresh that confirms it.
  const [token, setToken] = useState<string | null>(null)
  const [user, setUser] = useState<AuthUser | null>(null)
  const [isLoading, setIsLoading] = useState(true)

  // The server's clearAuthSessionCookie on logout is what actually ends the session.
  const clearSession = useCallback(() => {
    setToken(null)
    setUser(null)
  }, [])

  const setSession = useCallback((nextToken: string, nextUser?: AuthUser | null) => {
    const validToken = normalizeAuthToken(nextToken)
    setToken(validToken)
    setUser(validToken ? (nextUser ?? null) : null)
  }, [])

  const refreshSession = useCallback(async () => {
    setIsLoading(true)
    try {
      const result = await fetchCurrentAuthUser()
      if (result.clearLocalToken) {
        clearSession()
        return
      }
      setUser(result.user)
    } catch {
      notifyUnauthorized()
    } finally {
      setIsLoading(false)
    }
  }, [clearSession])

  useEffect(() => {
    void refreshSession()
  }, [refreshSession])

  useEffect(() => {
    const handleUnauthorized = () => {
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
        throw new Error(await readErrorMessage(response))
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
