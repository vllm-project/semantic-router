import { useEffect } from 'react'
import { Navigate, useParams } from 'react-router-dom'

import DashboardManagerLayout from '../components/DashboardManagerLayout'
import ContextCompressionPage from './ContextCompressionPage'

const PLUGIN_VIEW = 'context-compression'

export default function PluginOperationsPage() {
  const { plugin } = useParams<{ plugin?: string }>()

  useEffect(() => {
    document.title = 'Plugin Operations | vLLM Semantic Router'
  }, [])

  if (!plugin) {
    return <Navigate to={`/plugins/${PLUGIN_VIEW}`} replace />
  }
  if (plugin !== PLUGIN_VIEW) {
    return <Navigate to={`/plugins/${PLUGIN_VIEW}`} replace />
  }

  return (
    <DashboardManagerLayout
      eyebrow="Operate"
      title="Plugin Operations"
      description="Inspect and operate post-decision runtime plugins without promoting plugin-local controls into global platform pages."
      meta={[
        {
          label: 'Current plugin',
          value: 'Context Compression',
        },
        { label: 'Runtime layer', value: 'Post-decision processing' },
        { label: 'Scope', value: 'Live router' },
      ]}
      panelEyebrow="Plugin runtime"
      panelTitle="Operational plugin workspace"
      panelDescription="Inspect compression health, preview changes, and recover scoped state."
    >
      <ContextCompressionPage embedded />
    </DashboardManagerLayout>
  )
}
