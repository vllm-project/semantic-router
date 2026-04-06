import React, { useEffect, useState } from 'react'
import {
  BrowserRouter,
  Navigate,
  Outlet,
  Route,
  Routes,
  useLocation,
} from 'react-router-dom'
import LandingPage from './pages/LandingPage'
import MonitoringPage from './pages/MonitoringPage'
import PlaygroundPage from './pages/PlaygroundPage'
import PlaygroundFullscreenPage from './pages/PlaygroundFullscreenPage'
import TopologyPage from './pages/TopologyPage'
import TracingPage from './pages/TracingPage'
import StatusPage from './pages/StatusPage'
import LogsPage from './pages/LogsPage'
import EvaluationPage from './pages/EvaluationPage'
import MLSetupPage from './pages/MLSetupPage'
import RatingsPage from './pages/RatingsPage'
import BuilderPage from './pages/BuilderPage'
import DashboardPage from './pages/DashboardPage'
import FleetSimOverviewPage from './pages/FleetSimOverviewPage'
import FleetSimWorkloadsPage from './pages/FleetSimWorkloadsPage'
import FleetSimFleetsPage from './pages/FleetSimFleetsPage'
import FleetSimRunsPage from './pages/FleetSimRunsPage'
import OpenClawPage from './pages/OpenClawPage'
import UsersPage from './pages/UsersPage'
import InsightsPage from './pages/InsightsPage'
import InsightsRecordPage from './pages/InsightsRecordPage'
import KnowledgeMapPage from './pages/KnowledgeMapPage'
import { ConfigSection } from './components/ConfigNav'
import { ReadonlyProvider } from './contexts/ReadonlyContext'
import { SetupProvider, useSetup } from './contexts/SetupContext'
import { AuthProvider, useAuth } from './contexts/AuthContext'
import SetupWizardPage from './pages/SetupWizardPage'
import OnboardingGuide from './components/OnboardingGuide'
import LoginPage from './pages/LoginPage'
import AuthTransitionPage from './pages/AuthTransitionPage'
import { canAccessMLSetup } from './utils/accessControl'
import {
  ConfigSectionRoute,
  KnowledgeBaseRoute,
  LegacyTaxonomyRedirect,
  SetupStatusPage,
  ShellPageRoute,
} from './appShellRouteSupport'

const AuthGate: React.FC = () => {
  const { isAuthenticated, isLoading } = useAuth()
  const location = useLocation()

  if (isLoading) {
    return (
      <SetupStatusPage
        title="Authenticating"
        description="Checking session state..."
        actionLabel="Retry"
        onAction={() => {
          window.location.reload()
        }}
      />
    )
  }

  if (!isAuthenticated) {
    const from = `${location.pathname}${location.search}${location.hash}`
    return <Navigate to="/login" state={{ from }} replace />
  }

  return <Outlet />
}

const AuthenticatedShell: React.FC = () => {
  const { setupState } = useSetup()
  const location = useLocation()
  const isSetupMode = setupState?.setupMode ?? false

  if (isSetupMode && location.pathname !== '/setup') {
    return <Navigate to="/setup" replace />
  }

  if (!isSetupMode && location.pathname === '/setup') {
    return <Navigate to="/dashboard" replace />
  }

  return (
    <>
      <Outlet />
      {!isSetupMode && location.pathname !== '/setup' && <OnboardingGuide />}
    </>
  )
}

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
        <Route path="/" element={<LandingPage />} />
        <Route path="/login" element={<LoginPage />} />
        <Route path="/auth/transition" element={<AuthTransitionPage />} />

        <Route element={<AuthGate />}>
          <Route element={<AuthenticatedShell />}>
            <Route path="/setup" element={<SetupWizardPage />} />
            <Route
              path="/dashboard"
              element={
                <ShellPageRoute
                  configSection={configSection}
                  setConfigSection={setConfigSection}
                >
                  <DashboardPage />
                </ShellPageRoute>
              }
            />
            <Route
              path="/monitoring"
              element={
                <ShellPageRoute
                  configSection={configSection}
                  setConfigSection={setConfigSection}
                >
                  <MonitoringPage />
                </ShellPageRoute>
              }
            />
            <Route
              path="/config"
              element={
                <ConfigSectionRoute
                  configSection={configSection}
                  setConfigSection={setConfigSection}
                />
              }
            />
            <Route
              path="/config/:section"
              element={
                <ConfigSectionRoute
                  configSection={configSection}
                  setConfigSection={setConfigSection}
                />
              }
            />
            <Route path="/knowledge-bases" element={<Navigate to="/knowledge-bases/bases" replace />} />
            <Route
              path="/knowledge-bases/:name/map"
              element={<KnowledgeMapPage />}
            />
            <Route
              path="/knowledge-bases/:view"
              element={
                <KnowledgeBaseRoute
                  configSection={configSection}
                  setConfigSection={setConfigSection}
                />
              }
            />
            <Route path="/taxonomy" element={<Navigate to="/knowledge-bases/bases" replace />} />
            <Route path="/taxonomy/:view" element={<LegacyTaxonomyRedirect />} />
            <Route
              path="/playground"
              element={
                <ShellPageRoute
                  configSection={configSection}
                  setConfigSection={setConfigSection}
                  hideHeaderOnMobile={true}
                  hideAccountControl={true}
                >
                  <PlaygroundPage />
                </ShellPageRoute>
              }
            />
            <Route path="/playground/fullscreen" element={<PlaygroundFullscreenPage />} />
            <Route
              path="/topology"
              element={
                <ShellPageRoute
                  configSection={configSection}
                  setConfigSection={setConfigSection}
                >
                  <TopologyPage />
                </ShellPageRoute>
              }
            />
            <Route
              path="/tracing"
              element={
                <ShellPageRoute
                  configSection={configSection}
                  setConfigSection={setConfigSection}
                >
                  <TracingPage />
                </ShellPageRoute>
              }
            />
            <Route
              path="/status"
              element={
                <ShellPageRoute
                  configSection={configSection}
                  setConfigSection={setConfigSection}
                >
                  <StatusPage />
                </ShellPageRoute>
              }
            />
            <Route
              path="/logs"
              element={
                <ShellPageRoute
                  configSection={configSection}
                  setConfigSection={setConfigSection}
                >
                  <LogsPage />
                </ShellPageRoute>
              }
            />
            <Route
              path="/insights"
              element={
                <ShellPageRoute
                  configSection={configSection}
                  setConfigSection={setConfigSection}
                >
                  <InsightsPage />
                </ShellPageRoute>
              }
            />
            <Route
              path="/insights/:recordId"
              element={
                <ShellPageRoute
                  configSection={configSection}
                  setConfigSection={setConfigSection}
                >
                  <InsightsRecordPage />
                </ShellPageRoute>
              }
            />
            <Route
              path="/evaluation"
              element={
                <ShellPageRoute
                  configSection={configSection}
                  setConfigSection={setConfigSection}
                >
                  <EvaluationPage />
                </ShellPageRoute>
              }
            />
            <Route
              path="/ml-setup"
              element={
                canUseMLSetup ? (
                  <ShellPageRoute
                    configSection={configSection}
                    setConfigSection={setConfigSection}
                  >
                    <MLSetupPage />
                  </ShellPageRoute>
                ) : (
                  <Navigate to="/dashboard" replace />
                )
              }
            />
            <Route
              path="/ratings"
              element={
                <ShellPageRoute
                  configSection={configSection}
                  setConfigSection={setConfigSection}
                >
                  <RatingsPage />
                </ShellPageRoute>
              }
            />
            <Route
              path="/fleet-sim"
              element={
                <ShellPageRoute
                  configSection={configSection}
                  setConfigSection={setConfigSection}
                >
                  <FleetSimOverviewPage />
                </ShellPageRoute>
              }
            />
            <Route
              path="/fleet-sim/workloads"
              element={
                <ShellPageRoute
                  configSection={configSection}
                  setConfigSection={setConfigSection}
                >
                  <FleetSimWorkloadsPage />
                </ShellPageRoute>
              }
            />
            <Route
              path="/fleet-sim/fleets"
              element={
                <ShellPageRoute
                  configSection={configSection}
                  setConfigSection={setConfigSection}
                >
                  <FleetSimFleetsPage />
                </ShellPageRoute>
              }
            />
            <Route
              path="/fleet-sim/runs"
              element={
                <ShellPageRoute
                  configSection={configSection}
                  setConfigSection={setConfigSection}
                >
                  <FleetSimRunsPage />
                </ShellPageRoute>
              }
            />
            <Route
              path="/builder"
              element={
                <ShellPageRoute
                  configSection={configSection}
                  setConfigSection={setConfigSection}
                >
                  <BuilderPage />
                </ShellPageRoute>
              }
            />
            <Route
              path="/clawos"
              element={
                <ShellPageRoute
                  configSection={configSection}
                  setConfigSection={setConfigSection}
                >
                  <OpenClawPage />
                </ShellPageRoute>
              }
            />
            <Route
              path="/users"
              element={
                <ShellPageRoute
                  configSection={configSection}
                  setConfigSection={setConfigSection}
                >
                  <UsersPage />
                </ShellPageRoute>
              }
            />
            <Route path="/openclaw" element={<Navigate to="/clawos" replace />} />
            <Route path="*" element={<Navigate to={setupMode ? '/setup' : '/dashboard'} replace />} />
          </Route>
        </Route>
      </Routes>
    </BrowserRouter>
  )
}

const App: React.FC = () => {
  const [isInIframe, setIsInIframe] = useState(false)

  useEffect(() => {
    // Detect if we're running inside an iframe (potential loop)
    if (window.self !== window.top) {
      setIsInIframe(true)
      console.warn('Dashboard detected it is running inside an iframe - this may indicate a loop')
    }
  }, [])

  // If we're in an iframe, show a warning instead of rendering the full app
  if (isInIframe) {
    return (
      <div
        style={{
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          height: '100vh',
          padding: '2rem',
          textAlign: 'center',
          backgroundColor: 'var(--color-bg)',
          color: 'var(--color-text)',
        }}
      >
        <div style={{ fontSize: '4rem', marginBottom: '1rem' }}>⚠️</div>
        <h1 style={{ fontSize: '1.5rem', marginBottom: '1rem', color: 'var(--color-danger)' }}>
          Nested Dashboard Detected
        </h1>
        <p style={{ maxWidth: '600px', lineHeight: '1.6', color: 'var(--color-text-secondary)' }}>
          The dashboard has detected that it is running inside an iframe. This usually indicates a
          configuration error where the dashboard is trying to embed itself.
        </p>
        <p style={{ marginTop: '1rem', color: 'var(--color-text-secondary)' }}>
          Please check your Grafana dashboard path and backend proxy configuration.
        </p>
        <button
          onClick={() => {
            window.top?.location.reload()
          }}
          style={{
            marginTop: '1.5rem',
            padding: '0.75rem 1.5rem',
            backgroundColor: 'var(--color-primary)',
            color: 'white',
            border: 'none',
            borderRadius: 'var(--radius-md)',
            fontSize: '0.875rem',
            fontWeight: '500',
            cursor: 'pointer',
          }}
        >
          Open Dashboard in New Tab
        </button>
      </div>
    )
  }

  return (
    <AuthProvider>
      <ReadonlyProvider>
        <SetupProvider>
          <AppRouter />
        </SetupProvider>
      </ReadonlyProvider>
    </AuthProvider>
  )
}

export default App
