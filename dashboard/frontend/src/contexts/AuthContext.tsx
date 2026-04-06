import React, { createContext, ReactNode, useContext, useEffect, useState } from 'react'
import {
  installAuthenticatedFetch,
  UNAUTHORIZED_EVENT,
} from '../utils/authFetch'

interface AuthUser {
  id: string
  email: string
  name: string
  role?: string
  permissions?: string[]
}

interface AuthContextValue {
  user: AuthUser | null
  isLoading: boolean
  isAuthenticated: boolean
  login: (email: string, password: string) => Promise<void>
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
  const [user, setUser] = useState<AuthUser | null>(null)
  const [isLoading, setIsLoading] = useState(true)

  const clearSessionState = () => {
    setUser(null)
  }

  const refreshSession = async () => {
    setIsLoading(true)
    try {
      const response = await fetch('/api/auth/me')
      if (!response.ok) {
        if (response.status === 401) {
          clearSessionState()
        }
        return
      }
      const payload = (await response.json()) as { user?: AuthUser }
      setUser(payload?.user ?? null)
    } catch {
      clearSessionState()
    } finally {
      setIsLoading(false)
    }
  }

  useEffect(() => {
    void refreshSession()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  useEffect(() => {
    const handleUnauthorized = () => {
      clearSessionState()
      setIsLoading(false)
    }

    window.addEventListener(UNAUTHORIZED_EVENT, handleUnauthorized)
    return () => window.removeEventListener(UNAUTHORIZED_EVENT, handleUnauthorized)
  }, [])

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

      const payload = (await response.json()) as { user?: AuthUser }
      if (payload.user) {
        setUser(payload.user)
        return
      }

      await refreshSession()
    } finally {
      setIsLoading(false)
    }
  }

  const logout = () => {
    clearSessionState()
    setIsLoading(false)
    void fetch('/api/auth/logout', { method: 'POST' }).catch(() => undefined)
  }

  return (
    <AuthContext.Provider
      value={{
        user,
        isLoading,
        isAuthenticated: Boolean(user),
        login,
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
