import React, { useEffect, useState } from 'react'
import { BrowserRouter, Routes, Route } from 'react-router-dom'
import Layout from './components/Layout'
import LandingPage from './pages/LandingPage'
import MonitoringPage from './pages/MonitoringPage'
import ConfigPage from './pages/ConfigPage'
import PlaygroundPage from './pages/PlaygroundPage'

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
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<LandingPage />} />
        <Route path="/monitoring" element={<Layout><MonitoringPage /></Layout>} />
        <Route path="/config" element={<Layout><ConfigPage /></Layout>} />
        <Route path="/playground" element={<Layout><PlaygroundPage /></Layout>} />
      </Routes>
    </BrowserRouter>
  )
}

export default App
