import React, { useState } from 'react'
import { BrowserRouter, Route, Routes } from 'react-router-dom'
import type { ConfigSection } from '../components/ConfigNav'
import { useAuth } from '../contexts/AuthContext'
import { useSetup } from '../contexts/SetupContext'
import AuthTransitionPage from '../pages/AuthTransitionPage'
import { canAccessMLSetup } from '../utils/accessControl'
import AuthGate from './AuthGate'
import AuthenticatedShell from './AuthenticatedShell'
import { renderAuthenticatedAppRoutes } from './AuthenticatedAppRoutes'
import RecoverableLazyRoute from './RecoverableLazyRoute'
import SetupStatusPage from './SetupStatusPage'
import { loadLandingPage, loadLoginPage } from './routeLoaders'

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
        <Route
          path="/"
          element={<RecoverableLazyRoute loader={loadLandingPage} routeLabel="Landing page" />}
        />
        <Route
          path="/login"
          element={<RecoverableLazyRoute loader={loadLoginPage} routeLabel="Login" />}
        />
        <Route path="/auth/transition" element={<AuthTransitionPage />} />

        <Route element={<AuthGate />}>
          <Route element={<AuthenticatedShell />}>
            {renderAuthenticatedAppRoutes({
              configSection,
              setConfigSection,
              canUseMLSetup,
              user,
              setupMode,
            })}
          </Route>
        </Route>
      </Routes>
    </BrowserRouter>
  )
}

export default AppRouter
