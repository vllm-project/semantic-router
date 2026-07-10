import React, { Suspense, lazy, useState } from 'react'
import {
  BrowserRouter,
  Route,
  Routes,
} from 'react-router-dom'
import type { ConfigSection } from '../components/ConfigNav'
import { useAuth } from '../contexts/AuthContext'
import { useSetup } from '../contexts/SetupContext'
import AuthTransitionPage from '../pages/AuthTransitionPage'
import { canAccessMLSetup } from '../utils/accessControl'
import AuthGate from './AuthGate'
import AuthenticatedShell from './AuthenticatedShell'
import { renderAuthenticatedAppRoutes } from './AuthenticatedAppRoutes'
import RouteLoadingFallback from './RouteLoadingFallback'
import SetupStatusPage from './SetupStatusPage'

const LandingPage = lazy(() => import('../pages/LandingPage'))
const LoginPage = lazy(() => import('../pages/LoginPage'))

const renderLazyRoute = (element: React.ReactElement) => (
  <Suspense fallback={<RouteLoadingFallback />}>
    {element}
  </Suspense>
)

const AppRouter: React.FC = () => {
  const { setupState, isLoading, error, refreshSetupState } = useSetup()
  const { user } = useAuth()
  const [configSection, setConfigSection] = useState<ConfigSection>('global-config')
  const canUseMLSetup = canAccessMLSetup(user)

  if (isLoading) {
    return (
      <SetupStatusPage
        title="Loading setup state"
        description="The dashboard is checking whether this workspace is already activated or still in first-run setup mode."
        actionLabel="Refresh"
        variant="loading"
        onAction={() => {
          window.location.reload()
        }}
      />
    )
  }

  if (error) {
    return (
      <SetupStatusPage
        title="Unable to load setup state"
        description={error}
        actionLabel="Retry"
        onAction={() => {
          void refreshSetupState()
        }}
      />
    )
  }

  const setupMode = setupState?.setupMode ?? false

  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={renderLazyRoute(<LandingPage />)} />
        <Route path="/login" element={renderLazyRoute(<LoginPage />)} />
        <Route path="/auth/transition" element={<AuthTransitionPage />} />

        <Route element={<AuthGate />}>
          <Route element={<AuthenticatedShell />}>
            {renderAuthenticatedAppRoutes({
              configSection,
              setConfigSection,
              canUseMLSetup,
              setupMode,
            })}
          </Route>
        </Route>
      </Routes>
    </BrowserRouter>
  )
}

export default AppRouter
