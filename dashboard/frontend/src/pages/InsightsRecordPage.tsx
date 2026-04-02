import { useCallback, useEffect, useMemo, useState } from 'react'
import { useNavigate, useParams } from 'react-router-dom'

import ViewPanel, { type ViewPanelAction } from '../components/ViewPanel'
import { useReadonly } from '../contexts/ReadonlyContext'

import configStyles from './ConfigPage.module.css'
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
    if (!shareUrl || !navigator.clipboard?.writeText) {
      return
    }

    try {
      await navigator.clipboard.writeText(shareUrl)
      setCopyState('copied')
    } catch {
      setCopyState('idle')
    }
  }, [shareUrl])

  const panelActions = useMemo<ViewPanelAction[]>(
    () => [
      {
        label: copyState === 'copied' ? 'Link Copied' : 'Copy Link',
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
      description="Open, review, and share a single replay-backed request record."
      configArea="Analysis"
      scope="Shareable request detail"
      panelTitle="Semantic Router Record"
      panelDescription="A standalone deep-link view of one replay event, with the same detail body used in the in-page modal."
      pills={[
        { label: 'Record Detail', active: true },
        { label: 'Shareable Link' },
        { label: 'Replay Event' },
      ]}
    >
      {error ? (
        <div className={styles.error}>
          <span>{error}</span>
        </div>
      ) : null}

      <div className={configStyles.sectionPanel}>
        <section className={configStyles.sectionTableBlock}>
          <div className={styles.toolbar}>
            <div>
              <h2 className={styles.sectionTitle}>Record Detail</h2>
              <p className={styles.sectionSubtitle}>
                Share this page directly when you need a stable URL for a single request replay.
              </p>
            </div>
            <div className={styles.toolbarActions}>
              <button type="button" onClick={() => void loadRecord()} className={styles.refreshButton}>
                Refresh
              </button>
            </div>
          </div>

          {loading ? (
            <div className={styles.loading}>
              <div className={styles.spinner} />
              <p>Loading insight record...</p>
            </div>
          ) : null}

          {!loading && !error && record ? (
            <ViewPanel
              title={buildInsightsRecordTitle(record)}
              sections={buildInsightsRecordSections(record, { isReadonly })}
              onClose={() => navigate('/insights')}
              closeLabel="Back to Insights"
              actions={panelActions}
              variant="page"
            />
          ) : null}
        </section>
      </div>
    </ConfigPageManagerLayout>
  )
}
