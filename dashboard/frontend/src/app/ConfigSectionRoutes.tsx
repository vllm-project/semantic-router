import React, { useEffect } from 'react'
import { Navigate, useParams } from 'react-router-dom'
import type { KnowledgeBaseView } from '../pages/TaxonomyPage'
import ConfigPage from '../pages/ConfigPage'
import TaxonomyPage from '../pages/TaxonomyPage'
import AppShellLayout from './AppShellLayout'
import type { ConfigSection } from '../components/ConfigNav'

export const ConfigSectionRoute: React.FC<{
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
    <AppShellLayout configSection={configSection} setConfigSection={setConfigSection}>
      <ConfigPage activeSection={configSection} />
    </AppShellLayout>
  )
}

export const KnowledgeBaseRoute: React.FC<{
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
    <AppShellLayout configSection={configSection} setConfigSection={setConfigSection}>
      <TaxonomyPage activeView={activeView} />
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
