import React, { useEffect } from 'react'
import { Navigate, useParams } from 'react-router-dom'
import AppShellLayout from './AppShellLayout'
import type { ConfigSection } from '../components/ConfigNav'
import RecoverableLazyRoute from './RecoverableLazyRoute'
import { loadConfigPage } from './routeLoaders'
import { useAuth } from '../contexts/AuthContext'
import { canAccessDashboardPath } from '../utils/accessControl'

export const ConfigSectionRoute: React.FC<{
  configSection: ConfigSection
  setConfigSection: (section: ConfigSection) => void
}> = ({ configSection, setConfigSection }) => {
  const { user } = useAuth()
  const { section } = useParams<{ section: string }>()
  const normalized = section?.toLowerCase() ?? ''

  useEffect(() => {
    if (!section) {
      if (configSection !== 'models') {
        setConfigSection('models')
      }
      return
    }

    const sectionMap: Record<string, ConfigSection> = {
      signals: 'signals',
      projections: 'projections',
      decisions: 'decisions',
      models: 'models',
      'entrypoints-recipes': 'entrypoints-recipes',
      agent: 'agent',
    }

    const mapped = sectionMap[normalized]
    if (mapped && mapped !== configSection) {
      setConfigSection(mapped)
    }
  }, [section, normalized, configSection, setConfigSection])

  const supportedSections = new Set([
    'models',
    'entrypoints-recipes',
    'signals',
    'projections',
    'decisions',
    'agent',
  ])
  if (section && !supportedSections.has(normalized)) {
    return <Navigate to="/config/models" replace />
  }

  if (!canAccessDashboardPath(user, `/config/${normalized || 'models'}`)) {
    return <Navigate to="/dashboard" replace />
  }

  return (
    <AppShellLayout configSection={configSection} setConfigSection={setConfigSection}>
      <RecoverableLazyRoute
        loader={loadConfigPage}
        routeLabel="Configuration"
        componentProps={{ activeSection: configSection }}
      />
    </AppShellLayout>
  )
}
