import React, { useEffect, useState } from 'react'
import { BrowserRouter, Routes, Route, useParams } from 'react-router-dom'
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
import ReplayPage from './pages/ReplayPage'
import EvaluationPage from './pages/EvaluationPage'
import MLSetupPage from './pages/MLSetupPage'
import RatingsPage from './pages/RatingsPage'
import BuilderPage from './pages/BuilderPage'
import DashboardPage from './pages/DashboardPage'
import { ConfigSection } from './components/ConfigNav'
import { ReadonlyProvider } from './contexts/ReadonlyContext'

const ConfigSectionRoute: React.FC<{
  configSection: ConfigSection
  setConfigSection: (section: ConfigSection) => void
}> = ({ configSection, setConfigSection }) => {
  const { section } = useParams<{ section: string }>()

  useEffect(() => {
    if (!section) return

    const normalized = section.toLowerCase()
    const sectionMap: Record<string, ConfigSection> = {
      signals: 'signals',
      routes: 'decisions',
      decisions: 'decisions',
      endpoints: 'models',
      models: 'models',
    }

    const mapped = sectionMap[normalized]
    if (mapped && mapped !== configSection) {
      setConfigSection(mapped)
    }
  }, [section, configSection, setConfigSection])

  return (
    <Layout
      configSection={configSection}
      onConfigSectionChange={(nextSection) => setConfigSection(nextSection as ConfigSection)}
    >
      <ConfigPage activeSection={configSection} />
    </Layout>
  )
}

const App: React.FC = () => {
  const [isInIframe, setIsInIframe] = useState(false)
  const [configSection, setConfigSection] = useState<ConfigSection>('signals')

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
    <ReadonlyProvider>
      <BrowserRouter>
        <Routes>
          <Route
            path="/"
            element={<LandingPage />}
          />
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
          <Route
            path="/playground"
            element={
              <Layout
                configSection={configSection}
                onConfigSectionChange={(section) => setConfigSection(section as ConfigSection)}
                hideHeaderOnMobile={true}
              >
                <PlaygroundPage />
              </Layout>
            }
          />
          <Route
            path="/playground/fullscreen"
            element={<PlaygroundFullscreenPage />}
          />
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
            path="/replay"
            element={
              <Layout
                configSection={configSection}
                onConfigSectionChange={(section) => setConfigSection(section as ConfigSection)}
              >
                <ReplayPage />
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
              <Layout
                configSection={configSection}
                onConfigSectionChange={(section) => setConfigSection(section as ConfigSection)}
              >
                <MLSetupPage />
              </Layout>
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
        </Routes>
      </BrowserRouter>
    </ReadonlyProvider>
  )
}

export default App
