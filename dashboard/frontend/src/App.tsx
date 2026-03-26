import React, { useEffect, useState } from 'react'
import {
  BrowserRouter,
  Navigate,
  Outlet,
  Route,
  Routes,
  useLocation,
  useParams,
} from 'react-router-dom'
import Layout from './components/Layout'
import LandingPage from './pages/LandingPage'
import MonitoringPage from './pages/MonitoringPage'
import ConfigPage from './pages/ConfigPage'
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
import TaxonomyPage, { type KnowledgeBaseView } from './pages/TaxonomyPage'
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

const ConfigSectionRoute: React.FC<{
  configSection: ConfigSection
  setConfigSection: (section: ConfigSection) => void
}> = ({ configSection, setConfigSection }) => {
  const { section } = useParams<{ section: string }>()
  const normalized = section?.toLowerCase() ?? ''
  const redirectToKnowledgeBases =
    normalized === 'classifiers' ||
    normalized === 'taxonomy-classifiers' ||
    normalized === 'knowledge-bases' ||
    normalized === 'kbs'

  useEffect(() => {
    if (!section) {
      if (configSection !== 'global-config') {
        setConfigSection('global-config')
      }
      return
    }

    const sectionMap: Record<string, ConfigSection> = {
      global: 'global-config',
      'global-config': 'global-config',
      'router-config': 'global-config',
      signals: 'signals',
      projections: 'projections',
      routes: 'decisions',
      decisions: 'decisions',
      endpoints: 'models',
      models: 'models',
      mcp: 'mcp',
    }

    const mapped = sectionMap[normalized]
    if (mapped && mapped !== configSection) {
      setConfigSection(mapped)
    }
  }, [section, normalized, configSection, setConfigSection])

  if (redirectToKnowledgeBases) {
    return <Navigate to="/knowledge-bases/bases" replace />
  }

  return (
    <Layout
      configSection={configSection}
      onConfigSectionChange={(nextSection) => setConfigSection(nextSection as ConfigSection)}
    >
      <ConfigPage activeSection={configSection} />
    </Layout>
  )
}

const KnowledgeBaseRoute: React.FC<{
  configSection: ConfigSection
  setConfigSection: (section: ConfigSection) => void
}> = ({ configSection, setConfigSection }) => {
  const { view } = useParams<{ view: string }>()
  const normalized = (view?.toLowerCase() ?? 'bases') as KnowledgeBaseView
  const activeView: KnowledgeBaseView = ['bases', 'groups', 'labels'].includes(normalized)
    ? normalized
    : 'bases'

  if (view && activeView !== normalized) {
    return <Navigate to={`/knowledge-bases/${activeView}`} replace />
  }

  return (
    <Layout
      configSection={configSection}
      onConfigSectionChange={(nextSection) => setConfigSection(nextSection as ConfigSection)}
    >
      <TaxonomyPage activeView={activeView} />
    </Layout>
  )
}

const LegacyTaxonomyRedirect: React.FC = () => {
  const { view } = useParams<{ view: string }>()
  const normalized = view?.toLowerCase() ?? 'classifiers'
  const viewMap: Record<string, KnowledgeBaseView> = {
    classifiers: 'bases',
    bases: 'bases',
    'knowledge-bases': 'bases',
    tiers: 'groups',
    categories: 'labels',
    exemplars: 'labels',
  }
  const nextView = viewMap[normalized] ?? 'bases'
  return <Navigate to={`/knowledge-bases/${nextView}`} replace />
}

const SetupStatusPage: React.FC<{
  title: string
  description: string
  actionLabel: string
  onAction: () => void
}> = ({ title, description, actionLabel, onAction }) => (
  <div
    style={{
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      minHeight: '100vh',
      padding: '2rem',
      background:
        'radial-gradient(circle at top, rgba(118, 185, 0, 0.12), transparent 30%), var(--color-bg)',
    }}
  >
    <div
      style={{
        width: '100%',
        maxWidth: '560px',
        padding: '2rem',
        borderRadius: '1rem',
        border: '1px solid var(--color-border)',
        background: 'var(--color-bg-secondary)',
        boxShadow: '0 20px 48px rgba(0, 0, 0, 0.28)',
      }}
    >
      <h1 style={{ fontSize: '1.5rem', marginBottom: '0.75rem' }}>{title}</h1>
      <p style={{ color: 'var(--color-text-secondary)', lineHeight: '1.6' }}>{description}</p>
      <button
        onClick={onAction}
        style={{
          marginTop: '1.25rem',
          padding: '0.75rem 1.15rem',
          borderRadius: '0.75rem',
          background: 'var(--color-primary)',
          color: '#081000',
          fontWeight: 700,
        }}
      >
        {actionLabel}
      </button>
    </div>
  </div>
)

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
                <Layout
                  configSection={configSection}
                  onConfigSectionChange={(section) => setConfigSection(section as ConfigSection)}
                >
                  <DashboardPage />
                </Layout>
              }
            />
            <Route
              path="/monitoring"
              element={
                <Layout
                  configSection={configSection}
                  onConfigSectionChange={(section) => setConfigSection(section as ConfigSection)}
                >
                  <MonitoringPage />
                </Layout>
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
              element={
                <Layout
                  configSection={configSection}
                  onConfigSectionChange={(section) => setConfigSection(section as ConfigSection)}
                >
                  <KnowledgeMapPage />
                </Layout>
              }
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
                <Layout
                  configSection={configSection}
                  onConfigSectionChange={(section) => setConfigSection(section as ConfigSection)}
                  hideHeaderOnMobile={true}
                  hideAccountControl={true}
                >
                  <PlaygroundPage />
                </Layout>
              }
            />
            <Route path="/playground/fullscreen" element={<PlaygroundFullscreenPage />} />
            <Route
              path="/topology"
              element={
                <Layout
                  configSection={configSection}
                  onConfigSectionChange={(section) => setConfigSection(section as ConfigSection)}
                >
                  <TopologyPage />
                </Layout>
              }
            />
            <Route
              path="/tracing"
              element={
                <Layout
                  configSection={configSection}
                  onConfigSectionChange={(section) => setConfigSection(section as ConfigSection)}
                >
                  <TracingPage />
                </Layout>
              }
            />
            <Route
              path="/status"
              element={
                <Layout
                  configSection={configSection}
                  onConfigSectionChange={(section) => setConfigSection(section as ConfigSection)}
                >
                  <StatusPage />
                </Layout>
              }
            />
            <Route
              path="/logs"
              element={
                <Layout
                  configSection={configSection}
                  onConfigSectionChange={(section) => setConfigSection(section as ConfigSection)}
                >
                  <LogsPage />
                </Layout>
              }
            />
            <Route
              path="/insights"
              element={
                <Layout
                  configSection={configSection}
                  onConfigSectionChange={(section) => setConfigSection(section as ConfigSection)}
                >
                  <InsightsPage />
                </Layout>
              }
            />
            <Route
              path="/evaluation"
              element={
                <Layout
                  configSection={configSection}
                  onConfigSectionChange={(section) => setConfigSection(section as ConfigSection)}
                >
                  <EvaluationPage />
                </Layout>
              }
            />
            <Route
              path="/ml-setup"
              element={
                canUseMLSetup ? (
                  <Layout
                    configSection={configSection}
                    onConfigSectionChange={(section) => setConfigSection(section as ConfigSection)}
                  >
                    <MLSetupPage />
                  </Layout>
                ) : (
                  <Navigate to="/dashboard" replace />
                )
              }
            />
            <Route
              path="/ratings"
              element={
                <Layout
                  configSection={configSection}
                  onConfigSectionChange={(section) => setConfigSection(section as ConfigSection)}
                >
                  <RatingsPage />
                </Layout>
              }
            />
            <Route
              path="/fleet-sim"
              element={
                <Layout
                  configSection={configSection}
                  onConfigSectionChange={(section) => setConfigSection(section as ConfigSection)}
                >
                  <FleetSimOverviewPage />
                </Layout>
              }
            />
            <Route
              path="/fleet-sim/workloads"
              element={
                <Layout
                  configSection={configSection}
                  onConfigSectionChange={(section) => setConfigSection(section as ConfigSection)}
                >
                  <FleetSimWorkloadsPage />
                </Layout>
              }
            />
            <Route
              path="/fleet-sim/fleets"
              element={
                <Layout
                  configSection={configSection}
                  onConfigSectionChange={(section) => setConfigSection(section as ConfigSection)}
                >
                  <FleetSimFleetsPage />
                </Layout>
              }
            />
            <Route
              path="/fleet-sim/runs"
              element={
                <Layout
                  configSection={configSection}
                  onConfigSectionChange={(section) => setConfigSection(section as ConfigSection)}
                >
                  <FleetSimRunsPage />
                </Layout>
              }
            />
            <Route
              path="/builder"
              element={
                <Layout
                  configSection={configSection}
                  onConfigSectionChange={(section) => setConfigSection(section as ConfigSection)}
                >
                  <BuilderPage />
                </Layout>
              }
            />
            <Route
              path="/clawos"
              element={
                <Layout
                  configSection={configSection}
                  onConfigSectionChange={(section) => setConfigSection(section as ConfigSection)}
                >
                  <OpenClawPage />
                </Layout>
              }
            />
            <Route
              path="/users"
              element={
                <Layout
                  configSection={configSection}
                  onConfigSectionChange={(section) => setConfigSection(section as ConfigSection)}
                >
                  <UsersPage />
                </Layout>
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
