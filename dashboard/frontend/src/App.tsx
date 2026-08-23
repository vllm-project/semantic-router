import React, { useEffect, useState } from 'react'
import { ReadonlyProvider } from './contexts/ReadonlyContext'
import { AuthProvider } from './contexts/AuthContext'
import AppRouter from './app/AppRouter'
import ProductIcon from './components/ProductIcon'

const App: React.FC = () => {
  const [isInIframe, setIsInIframe] = useState(false)

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', 'dark')
    document.documentElement.setAttribute('data-design', 'graphite')
    document.documentElement.style.colorScheme = 'dark'

    if (window.self !== window.top) {
      setIsInIframe(true)
      console.warn('Dashboard detected it is running inside an iframe - this may indicate a loop')
    }
  }, [])

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
        <ProductIcon
          name="alert"
          aria-hidden="true"
          style={{ width: '3rem', height: '3rem', marginBottom: '1rem' }}
        />
        <h1 style={{ fontSize: '1.5rem', marginBottom: '1rem', color: 'var(--color-danger)' }}>
          Nested Dashboard Detected
        </h1>
        <p style={{ maxWidth: '600px', lineHeight: '1.6', color: 'var(--color-text-secondary)' }}>
          This page cannot open inside another dashboard window.
        </p>
        <p style={{ marginTop: '1rem', color: 'var(--color-text-secondary)' }}>
          Open it directly to continue.
        </p>
        <button
          onClick={() => {
            window.top?.location.reload()
          }}
          style={{
            marginTop: '1.5rem',
            padding: '0.75rem 1.5rem',
            backgroundColor: 'var(--color-primary)',
            color: '#09090a',
            border: 'none',
            borderRadius: 'var(--radius-md)',
            fontSize: '0.875rem',
            fontWeight: '500',
            cursor: 'pointer',
          }}
        >
          Open dashboard
        </button>
      </div>
    )
  }

  return (
    <AuthProvider>
      <ReadonlyProvider>
        <AppRouter />
      </ReadonlyProvider>
    </AuthProvider>
  )
}

export default App
