import React, {
  createContext,
  ReactNode,
  useCallback,
  useContext,
  useEffect,
  useRef,
  useState,
} from 'react'
import {
  installAuthenticatedFetch,
  notifyUnauthorized,
  UNAUTHORIZED_EVENT,
} from '../utils/authFetch'
import {
  assertManagementMe,
  managementOperationRequest,
  setManagementNamespace,
} from '../utils/managementApiContract'
import { clearInvitationOnboarding } from '../utils/invitationOnboarding'
import { fetchCurrentAuthUser, hasAuthenticatedSession, type AuthUser } from './authSession'

interface AuthContextValue {
  user: AuthUser | null
  isLoading: boolean
  isAuthenticated: boolean
  login: (email: string, password: string) => Promise<void>
  setSession: (user?: AuthUser | null) => void
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
  const sessionGeneration = useRef(0)

  const clearSession = useCallback(() => {
    sessionGeneration.current += 1
    clearInvitationOnboarding()
    setManagementNamespace(null)
    setUser(null)
  }, [])

  const attachManagementIdentity = useCallback(async (nextUser: AuthUser | null) => {
    if (!nextUser) return null
    setManagementNamespace(null)
    try {
      const identity = assertManagementMe(
        await managementOperationRequest('getMe', { namespace: null }),
      )
      const namespace = identity.namespaces.find((scope) => scope.namespace.status === 'active')
      if (!namespace) {
        throw new Error('No active Router namespace is available for this account.')
      }
      setManagementNamespace(namespace.namespace.namespaceId)
      return {
        ...nextUser,
        managementIdentityStatus: 'ready' as const,
        managementIdentityError: undefined,
        managementPrincipalId: identity.principal.principalId,
        managementNamespaceId: namespace.namespace.namespaceId,
        managementUserId: namespace.user?.userId,
        managementTeams: namespace.teams,
        managementSelfServicePolicy: namespace.selfServicePolicy,
        managementPermissions: [
          ...new Set([
            ...(Array.isArray(identity.clusterPermissions) ? identity.clusterPermissions : []),
            ...(Array.isArray(namespace.permissions) ? namespace.permissions : []),
          ]),
        ],
        managementScopes: identity.namespaces.map((scope) => scope.namespace),
      }
    } catch (cause) {
      // Local Dashboard access may remain available, but every Router-owned
      // control and inference surface fails closed without scoped permissions.
      const detail =
        cause instanceof Error && cause.message.trim()
          ? cause.message.trim()
          : 'Router Management identity is unavailable.'
      return {
        ...nextUser,
        managementIdentityStatus: 'error' as const,
        managementIdentityError: detail,
        managementPrincipalId: undefined,
        managementNamespaceId: undefined,
        managementUserId: undefined,
        managementTeams: [],
        managementSelfServicePolicy: undefined,
        managementPermissions: [],
        managementScopes: [],
      }
    }
  }, [])

  const setSession = useCallback(
    (nextUser?: AuthUser | null) => {
      const generation = sessionGeneration.current + 1
      sessionGeneration.current = generation
      setManagementNamespace(null)
      if (!nextUser) {
        setUser(null)
        setIsLoading(false)
        return
      }
      setIsLoading(true)
      void attachManagementIdentity(nextUser).then((attached) => {
        if (sessionGeneration.current !== generation) return
        setUser(attached)
        setIsLoading(false)
      })
    },
    [attachManagementIdentity],
  )

  const refreshSession = useCallback(async () => {
    setIsLoading(true)
    try {
      const result = await fetchCurrentAuthUser()
      if (result.unauthorized) {
        clearSession()
        return
      }
      const generation = sessionGeneration.current + 1
      sessionGeneration.current = generation
      const nextUser = await attachManagementIdentity(result.user)
      if (sessionGeneration.current === generation) setUser(nextUser)
    } catch {
      notifyUnauthorized()
    } finally {
      setIsLoading(false)
    }
  }, [attachManagementIdentity, clearSession])

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

      const payload = (await response.json()) as { user?: AuthUser }
      if (!payload.user) throw new Error('Login response did not include a user session')
      setSession(payload.user)
    } catch (cause) {
      setIsLoading(false)
      throw cause
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
        user,
        isLoading,
        isAuthenticated: hasAuthenticatedSession(user),
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
