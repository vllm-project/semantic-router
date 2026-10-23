import React from 'react'
import { Navigate, useParams } from 'react-router-dom'
import AppShellLayout from './AppShellLayout'
import type { ConfigSection } from '../components/ConfigNav'
import RecoverableLazyRoute from './RecoverableLazyRoute'
import { loadConfigPage } from './routeLoaders'
import { useAuth } from '../contexts/AuthContext'
import { canAccessDashboardPath } from '../utils/accessControl'
import { normalizeConfigSection } from '../components/LayoutNavSupport'

export const ConfigSectionRoute: React.FC = () => {
  const { user } = useAuth()
  const { section } = useParams<{ section: string }>()
  const normalized = section?.toLowerCase() ?? ''
  if (!canAccessDashboardPath(user, normalized ? `/config/${normalized}` : '/config')) {
    return <Navigate to="/dashboard" replace />
  }

  const activeSection: ConfigSection | undefined = section
    ? normalizeConfigSection(normalized)
    : 'global-config'
  if (!activeSection) {
    return <Navigate to="/config/global-config" replace />
  }
  if (!section || normalized !== activeSection) {
    return <Navigate to={`/config/${activeSection}`} replace />
  }

  return (
    <AppShellLayout>
      <RecoverableLazyRoute
        loader={loadConfigPage}
        routeLabel="Configuration"
        componentProps={{ activeSection }}
      />
    </AppShellLayout>
  )
}
