import React from 'react'
import { Navigate, useParams } from 'react-router-dom'
import type { KnowledgeBaseView } from '../pages/TaxonomyPage'
import AppShellLayout from './AppShellLayout'
import type { ConfigSection } from '../components/ConfigNav'
import RecoverableLazyRoute from './RecoverableLazyRoute'
import { loadConfigPage, loadTaxonomyPage } from './routeLoaders'
import { useAuth } from '../contexts/AuthContext'
import { canAccessDashboardPath } from '../utils/accessControl'
import { normalizeConfigSection } from '../components/LayoutNavSupport'

export const ConfigSectionRoute: React.FC = () => {
  const { user } = useAuth()
  const { section } = useParams<{ section: string }>()
  const normalized = section?.toLowerCase() ?? ''
  const redirectToKnowledgeBases =
    normalized === 'classifiers' ||
    normalized === 'taxonomy-classifiers' ||
    normalized === 'knowledge-bases' ||
    normalized === 'kbs'

  if (redirectToKnowledgeBases) {
    return <Navigate to="/knowledge-bases/bases" replace />
  }

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

export const KnowledgeBaseRoute: React.FC = () => {
  const { user } = useAuth()
  const { view } = useParams<{ view: string }>()
  const normalized = (view?.toLowerCase() ?? 'bases') as KnowledgeBaseView
  const activeView: KnowledgeBaseView = ['bases', 'groups', 'labels'].includes(normalized)
    ? normalized
    : 'bases'

  if (view && activeView !== normalized) {
    return <Navigate to={`/knowledge-bases/${activeView}`} replace />
  }

  if (!canAccessDashboardPath(user, `/knowledge-bases/${activeView}`)) {
    return <Navigate to="/dashboard" replace />
  }

  return (
    <AppShellLayout>
      <RecoverableLazyRoute
        loader={loadTaxonomyPage}
        routeLabel="Knowledge bases"
        componentProps={{ activeView }}
      />
    </AppShellLayout>
  )
}

export const LegacyTaxonomyRedirect: React.FC = () => {
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
