import React, { createContext, ReactNode, useContext, useEffect, useState } from 'react'

interface AuthUser {
  id: string
  email: string
  name: string
  role?: string
}

interface AuthContextValue {
  token: string | null
  user: AuthUser | null
  isLoading: boolean
  isAuthenticated: boolean
  login: (email: string, password: string) => Promise<void>
  logout: () => void
  refreshSession: () => Promise<void>
}

const STORAGE_KEY = 'vsr_auth_token'

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

export const AuthProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  const [token, setToken] = useState<string | null>(localStorage.getItem(STORAGE_KEY))
  const [user, setUser] = useState<AuthUser | null>(null)
  const [isLoading, setIsLoading] = useState(false)

  useEffect(() => {
    if (!token) {
      setUser(null)
      return
    }

    void refreshSession()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const refreshSession = async () => {
    if (!token) {
      setUser(null)
      return
    }

    setIsLoading(true)
    try {
      const response = await fetch('/api/auth/me', {
        headers: {
          Authorization: `Bearer ${token}`,
        },
      })
      if (!response.ok) {
        if (response.status === 401) {
          setToken(null)
          setUser(null)
          localStorage.removeItem(STORAGE_KEY)
        }
        return
      }
      const payload = (await response.json()) as { user?: AuthUser }
      setUser(payload?.user ?? null)
    } catch {
      setUser(null)
      setToken(null)
      localStorage.removeItem(STORAGE_KEY)
    } finally {
      setIsLoading(false)
    }
  }

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
      localStorage.setItem(STORAGE_KEY, payload.token)
      setToken(payload.token)
      setUser(payload.user ?? null)
    } finally {
      setIsLoading(false)
    }
  }

  const logout = () => {
    setToken(null)
    setUser(null)
    localStorage.removeItem(STORAGE_KEY)
  }

  return (
    <AuthContext.Provider
      value={{
        token,
        user,
        isLoading,
        isAuthenticated: Boolean(token),
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
