import React, { useEffect } from 'react'
import { Navigate, useParams } from 'react-router-dom'
import Layout from './components/Layout'
import { ConfigSection } from './components/ConfigNav'
import ConfigPage from './pages/ConfigPage'
import TaxonomyPage, { type KnowledgeBaseView } from './pages/TaxonomyPage'

interface ShellPageRouteProps {
  children: React.ReactNode
  configSection: ConfigSection
  setConfigSection: (section: ConfigSection) => void
  hideHeaderOnMobile?: boolean
  hideAccountControl?: boolean
}

interface ConfigSectionRouteProps {
  configSection: ConfigSection
  setConfigSection: (section: ConfigSection) => void
}

export const ShellPageRoute: React.FC<ShellPageRouteProps> = ({
  children,
  configSection,
  setConfigSection,
  hideHeaderOnMobile = false,
  hideAccountControl = false,
}) => (
  <Layout
    configSection={configSection}
    onConfigSectionChange={(section) => setConfigSection(section as ConfigSection)}
    hideHeaderOnMobile={hideHeaderOnMobile}
    hideAccountControl={hideAccountControl}
  >
    {children}
  </Layout>
)

export const ConfigSectionRoute: React.FC<ConfigSectionRouteProps> = ({
  configSection,
  setConfigSection,
}) => {
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
    <ShellPageRoute
      configSection={configSection}
      setConfigSection={setConfigSection}
    >
      <ConfigPage activeSection={configSection} />
    </ShellPageRoute>
  )
}

export const KnowledgeBaseRoute: React.FC<ConfigSectionRouteProps> = ({
  configSection,
  setConfigSection,
}) => {
  const { view } = useParams<{ view: string }>()
  const normalized = (view?.toLowerCase() ?? 'bases') as KnowledgeBaseView
  const activeView: KnowledgeBaseView = ['bases', 'groups', 'labels'].includes(
    normalized,
  )
    ? normalized
    : 'bases'

  if (view && activeView !== normalized) {
    return <Navigate to={`/knowledge-bases/${activeView}`} replace />
  }

  return (
    <ShellPageRoute
      configSection={configSection}
      setConfigSection={setConfigSection}
    >
      <TaxonomyPage activeView={activeView} />
    </ShellPageRoute>
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

interface SetupStatusPageProps {
  title: string
  description: string
  actionLabel: string
  onAction: () => void
}

export const SetupStatusPage: React.FC<SetupStatusPageProps> = ({
  title,
  description,
  actionLabel,
  onAction,
}) => (
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
      <p style={{ color: 'var(--color-text-secondary)', lineHeight: '1.6' }}>
        {description}
      </p>
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
