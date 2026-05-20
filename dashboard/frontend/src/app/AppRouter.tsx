import React, { useState } from 'react'
import {
  BrowserRouter,
  Navigate,
  Route,
  Routes,
} from 'react-router-dom'
import LandingPage from '../pages/LandingPage'
import MonitoringPage from '../pages/MonitoringPage'
import PlaygroundPage from '../pages/PlaygroundPage'
import PlaygroundFullscreenPage from '../pages/PlaygroundFullscreenPage'
import TopologyPage from '../pages/TopologyPage'
import TracingPage from '../pages/TracingPage'
import StatusPage from '../pages/StatusPage'
import LogsPage from '../pages/LogsPage'
import EvaluationPage from '../pages/EvaluationPage'
import MLSetupPage from '../pages/MLSetupPage'
import RatingsPage from '../pages/RatingsPage'
import BuilderPage from '../pages/BuilderPage'
import DashboardPage from '../pages/DashboardPage'
import FleetSimOverviewPage from '../pages/FleetSimOverviewPage'
import FleetSimWorkloadsPage from '../pages/FleetSimWorkloadsPage'
import FleetSimFleetsPage from '../pages/FleetSimFleetsPage'
import FleetSimRunsPage from '../pages/FleetSimRunsPage'
import OpenClawPage from '../pages/OpenClawPage'
import UsersPage from '../pages/UsersPage'
import SecurityPolicyPage from '../pages/SecurityPolicyPage'
import InsightsPage from '../pages/InsightsPage'
import InsightsRecordPage from '../pages/InsightsRecordPage'
import KnowledgeMapPage from '../pages/KnowledgeMapPage'
import type { ConfigSection } from '../components/ConfigNav'
import { useAuth } from '../contexts/AuthContext'
import { useSetup } from '../contexts/SetupContext'
import SetupWizardPage from '../pages/SetupWizardPage'
import LoginPage from '../pages/LoginPage'
import AuthTransitionPage from '../pages/AuthTransitionPage'
import { canAccessMLSetup } from '../utils/accessControl'
import AppShellLayout from './AppShellLayout'
import AuthGate from './AuthGate'
import AuthenticatedShell from './AuthenticatedShell'
import {
  ConfigSectionRoute,
  KnowledgeBaseRoute,
  LegacyTaxonomyRedirect,
} from './ConfigSectionRoutes'
import SetupStatusPage from './SetupStatusPage'

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
              element={(
                <AppShellLayout configSection={configSection} setConfigSection={setConfigSection}>
                  <DashboardPage />
                </AppShellLayout>
              )}
            />
            <Route
              path="/monitoring"
              element={(
                <AppShellLayout configSection={configSection} setConfigSection={setConfigSection}>
                  <MonitoringPage />
                </AppShellLayout>
              )}
            />
            <Route
              path="/config"
              element={(
                <ConfigSectionRoute
                  configSection={configSection}
                  setConfigSection={setConfigSection}
                />
              )}
            />
            <Route
              path="/config/:section"
              element={(
                <ConfigSectionRoute
                  configSection={configSection}
                  setConfigSection={setConfigSection}
                />
              )}
            />
            <Route path="/knowledge-bases" element={<Navigate to="/knowledge-bases/bases" replace />} />
            <Route
              path="/knowledge-bases/:name/map"
              element={<KnowledgeMapPage />}
            />
            <Route
              path="/knowledge-bases/:view"
              element={(
                <KnowledgeBaseRoute
                  configSection={configSection}
                  setConfigSection={setConfigSection}
                />
              )}
            />
            <Route path="/taxonomy" element={<Navigate to="/knowledge-bases/bases" replace />} />
            <Route path="/taxonomy/:view" element={<LegacyTaxonomyRedirect />} />
            <Route
              path="/playground"
              element={(
                <AppShellLayout
                  configSection={configSection}
                  setConfigSection={setConfigSection}
                  hideHeaderOnMobile={true}
                  hideAccountControl={true}
                >
                  <PlaygroundPage />
                </AppShellLayout>
              )}
            />
            <Route path="/playground/fullscreen" element={<PlaygroundFullscreenPage />} />
            <Route
              path="/topology"
              element={(
                <AppShellLayout configSection={configSection} setConfigSection={setConfigSection}>
                  <TopologyPage />
                </AppShellLayout>
              )}
            />
            <Route
              path="/tracing"
              element={(
                <AppShellLayout configSection={configSection} setConfigSection={setConfigSection}>
                  <TracingPage />
                </AppShellLayout>
              )}
            />
            <Route
              path="/status"
              element={(
                <AppShellLayout configSection={configSection} setConfigSection={setConfigSection}>
                  <StatusPage />
                </AppShellLayout>
              )}
            />
            <Route
              path="/logs"
              element={(
                <AppShellLayout configSection={configSection} setConfigSection={setConfigSection}>
                  <LogsPage />
                </AppShellLayout>
              )}
            />
            <Route
              path="/insights"
              element={(
                <AppShellLayout configSection={configSection} setConfigSection={setConfigSection}>
                  <InsightsPage />
                </AppShellLayout>
              )}
            />
            <Route
              path="/insights/:recordId"
              element={(
                <AppShellLayout configSection={configSection} setConfigSection={setConfigSection}>
                  <InsightsRecordPage />
                </AppShellLayout>
              )}
            />
            <Route
              path="/evaluation"
              element={(
                <AppShellLayout configSection={configSection} setConfigSection={setConfigSection}>
                  <EvaluationPage />
                </AppShellLayout>
              )}
            />
            <Route
              path="/ml-setup"
              element={(
                canUseMLSetup ? (
                  <AppShellLayout configSection={configSection} setConfigSection={setConfigSection}>
                    <MLSetupPage />
                  </AppShellLayout>
                ) : (
                  <Navigate to="/dashboard" replace />
                )
              )}
            />
            <Route
              path="/ratings"
              element={(
                <AppShellLayout configSection={configSection} setConfigSection={setConfigSection}>
                  <RatingsPage />
                </AppShellLayout>
              )}
            />
            <Route
              path="/fleet-sim"
              element={(
                <AppShellLayout configSection={configSection} setConfigSection={setConfigSection}>
                  <FleetSimOverviewPage />
                </AppShellLayout>
              )}
            />
            <Route
              path="/fleet-sim/workloads"
              element={(
                <AppShellLayout configSection={configSection} setConfigSection={setConfigSection}>
                  <FleetSimWorkloadsPage />
                </AppShellLayout>
              )}
            />
            <Route
              path="/fleet-sim/fleets"
              element={(
                <AppShellLayout configSection={configSection} setConfigSection={setConfigSection}>
                  <FleetSimFleetsPage />
                </AppShellLayout>
              )}
            />
            <Route
              path="/fleet-sim/runs"
              element={(
                <AppShellLayout configSection={configSection} setConfigSection={setConfigSection}>
                  <FleetSimRunsPage />
                </AppShellLayout>
              )}
            />
            <Route
              path="/builder"
              element={(
                <AppShellLayout configSection={configSection} setConfigSection={setConfigSection}>
                  <BuilderPage />
                </AppShellLayout>
              )}
            />
            <Route
              path="/clawos"
              element={(
                <AppShellLayout configSection={configSection} setConfigSection={setConfigSection}>
                  <OpenClawPage />
                </AppShellLayout>
              )}
            />
            <Route
              path="/users"
              element={(
                <AppShellLayout configSection={configSection} setConfigSection={setConfigSection}>
                  <UsersPage />
                </AppShellLayout>
              )}
            />
            <Route
              path="/security"
              element={(
                <AppShellLayout configSection={configSection} setConfigSection={setConfigSection}>
                  <SecurityPolicyPage />
                </AppShellLayout>
              )}
            />
            <Route path="/openclaw" element={<Navigate to="/clawos" replace />} />
            <Route path="*" element={<Navigate to={setupMode ? '/setup' : '/dashboard'} replace />} />
          </Route>
        </Route>
      </Routes>
    </BrowserRouter>
  )
}

export default AppRouter
