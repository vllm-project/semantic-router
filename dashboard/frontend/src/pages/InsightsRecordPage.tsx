import { useCallback, useEffect, useMemo, useState } from 'react'
import { useNavigate, useParams } from 'react-router-dom'

import ViewPanel, { type ViewPanelAction } from '../components/ViewPanel'
import ProductIcon from '../components/ProductIcon'
import ProductLoadingState from '../components/ProductLoadingState'
import { useAuth } from '../contexts/AuthContext'
import { useReadonly } from '../contexts/ReadonlyContext'
import { canAccessReplayFlowDetails } from '../utils/accessControl'
import { copyText } from '../utils/clipboard'

import ConfigPageManagerLayout from './ConfigPageManagerLayout'
import styles from './InsightsPage.module.css'
import { fetchInsightsRecord } from './insightsPageApi'
import {
  buildInsightsRecordSections,
  buildInsightsRecordTitle,
  getInsightsRecordPath,
} from './insightsPageSupport'
import type { InsightsRecord } from './insightsPageTypes'

export default function InsightsRecordPage() {
  const navigate = useNavigate()
  const { recordId } = useParams<{ recordId: string }>()
  const { user } = useAuth()
  const { isReadonly } = useReadonly()
  const [record, setRecord] = useState<InsightsRecord | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [copyState, setCopyState] = useState<'idle' | 'copied'>('idle')

  const loadRecord = useCallback(async () => {
    if (!recordId) {
      setRecord(null)
      setError('Missing insight record ID.')
      setLoading(false)
      return
    }

    setLoading(true)

    try {
      const nextRecord = await fetchInsightsRecord(recordId)
      setRecord(nextRecord)
      setError(null)
    } catch (err) {
      setRecord(null)
      setError(err instanceof Error ? err.message : 'Unknown error')
    } finally {
      setLoading(false)
    }
  }, [recordId])

  useEffect(() => {
    void loadRecord()
  }, [loadRecord])

  useEffect(() => {
    if (copyState !== 'copied') {
      return undefined
    }

    const timeout = window.setTimeout(() => {
      setCopyState('idle')
    }, 2000)

    return () => window.clearTimeout(timeout)
  }, [copyState])

  const shareUrl = useMemo(() => {
    if (!recordId) {
      return ''
    }

    return `${window.location.origin}${getInsightsRecordPath(recordId)}`
  }, [recordId])

  const handleCopyLink = useCallback(async () => {
    if (!shareUrl) {
      return
    }

    if (await copyText(shareUrl)) {
      setCopyState('copied')
    } else {
      setCopyState('idle')
    }
  }, [shareUrl])

  const panelActions = useMemo<ViewPanelAction[]>(
    () => [
      {
        label: copyState === 'copied' ? 'Link copied' : 'Copy link',
        icon: copyState === 'copied' ? 'check' : 'copy',
        onClick: () => {
          void handleCopyLink()
        },
        tone: 'primary',
      },
    ],
    [copyState, handleCopyLink],
  )

  return (
    <ConfigPageManagerLayout
      eyebrow="Insights"
      title="Insight Record"
      description="Everything behind one routed request."
    >
      {error ? (
        <div className={styles.error} role="alert">
          <span>{error}</span>
        </div>
      ) : null}

      <section className={styles.recordStage} aria-label="Insight record" aria-busy={loading}>
        <div className={styles.recordStageHeader}>
          <button type="button" className={styles.backButton} onClick={() => navigate('/insights')}>
            <ProductIcon name="arrow-left" aria-hidden="true" />
            All requests
          </button>
          <button
            type="button"
            onClick={() => void loadRecord()}
            className={styles.refreshButton}
            disabled={loading}
          >
            <ProductIcon name="refresh" aria-hidden="true" />
            {loading ? 'Refreshing…' : 'Refresh'}
          </button>
        </div>
        {loading ? (
          <ProductLoadingState compact label="Loading insight record" />
        ) : null}

        {!loading && !error && record ? (
          <ViewPanel
            title={buildInsightsRecordTitle(record)}
            sections={buildInsightsRecordSections(record, {
              isReadonly,
              canViewReplayFlowDetails: canAccessReplayFlowDetails(user),
            })}
            onClose={() => navigate('/insights')}
            closeLabel="Back to Insights"
            actions={panelActions}
            variant="page"
          />
        ) : null}
      </section>
    </ConfigPageManagerLayout>
  )
}
