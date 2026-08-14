import { useEffect } from 'react'
import { Navigate, useNavigate, useParams } from 'react-router-dom'

import DashboardManagerLayout from '../components/DashboardManagerLayout'
import ContextCompressionPage from './ContextCompressionPage'
import ResponseCachePage from './ResponseCachePage'

type PluginView = 'response-cache' | 'context-compression'

const PLUGIN_VIEWS: ReadonlyArray<{ id: PluginView; label: string }> = [
  { id: 'response-cache', label: 'Response Cache' },
  { id: 'context-compression', label: 'Context Compression' },
]

export default function PluginOperationsPage() {
  const navigate = useNavigate()
  const { plugin } = useParams<{ plugin?: string }>()
  const activePlugin = PLUGIN_VIEWS.some((item) => item.id === plugin)
    ? (plugin as PluginView)
    : null

  useEffect(() => {
    document.title = 'Plugin Operations | vLLM Semantic Router'
  }, [])

  if (!plugin) {
    return <Navigate to="/plugins/response-cache" replace />
  }
  if (!activePlugin) {
    return <Navigate to="/plugins/response-cache" replace />
  }

  return (
    <DashboardManagerLayout
      eyebrow="Operate"
      title="Plugin Operations"
      description="Inspect and operate post-decision runtime plugins without promoting plugin-local controls into global platform pages."
      meta={[
        {
          label: 'Current plugin',
          value: PLUGIN_VIEWS.find((item) => item.id === activePlugin)?.label,
        },
        { label: 'Runtime layer', value: 'Post-decision processing' },
        { label: 'Scope', value: 'Live router' },
      ]}
      panelEyebrow="Plugin runtime"
      panelTitle="Operational plugin workspace"
      panelDescription="Cache and compression share one operational surface while retaining isolated capabilities, health, preview, and recovery controls."
      pills={PLUGIN_VIEWS.map((item) => ({
        label: item.label,
        active: item.id === activePlugin,
        onClick: () => navigate(`/plugins/${item.id}`),
      }))}
    >
      {activePlugin === 'response-cache' ? <ResponseCachePage embedded /> : null}
      {activePlugin === 'context-compression' ? <ContextCompressionPage embedded /> : null}
    </DashboardManagerLayout>
  )
}
