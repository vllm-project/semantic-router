import React, { useState } from 'react'
import { BrowserRouter, Route, Routes } from 'react-router-dom'
import type { ConfigSection } from '../components/ConfigNav'
import { useAuth } from '../contexts/AuthContext'
import AuthTransitionPage from '../pages/AuthTransitionPage'
import { canAccessMLSetup } from '../utils/accessControl'
import AuthGate from './AuthGate'
import AuthenticatedShell from './AuthenticatedShell'
import { renderAuthenticatedAppRoutes } from './AuthenticatedAppRoutes'
import RecoverableLazyRoute from './RecoverableLazyRoute'
import { loadLandingPage, loadLoginPage } from './routeLoaders'

const AppRouter: React.FC = () => {
  const { user } = useAuth()
  const [configSection, setConfigSection] = useState<ConfigSection>('models')
  const canUseMLSetup = canAccessMLSetup(user)

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
            })}
          </Route>
        </Route>
      </Routes>
    </BrowserRouter>
  )
}

export default AppRouter
