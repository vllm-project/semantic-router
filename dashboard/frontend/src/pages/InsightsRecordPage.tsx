import { useCallback, useEffect, useMemo, useState } from 'react'
import { useNavigate, useParams } from 'react-router-dom'

import ProductLoadingState from '../components/ProductLoadingState'
import ProductIcon from '../components/ProductIcon'
import { useAuth } from '../contexts/AuthContext'
import { useReadonly } from '../contexts/ReadonlyContext'
import { canAccessReplayFlowDetails } from '../utils/accessControl'

import styles from './InsightsPage.module.css'
import { fetchInsightsRecord } from './insightsPageApi'
import {
  buildInsightsRecordSections,
  buildInsightsRecordTitle,
  getInsightsLifecyclePresentation,
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

  const sections = useMemo(
    () =>
      record
        ? buildInsightsRecordSections(record, {
            isReadonly,
            canViewReplayFlowDetails: canAccessReplayFlowDetails(user),
          })
        : [],
    [isReadonly, record, user],
  )

  const lifecycle = record ? getInsightsLifecyclePresentation(record) : null

  return (
    <main className={styles.recordPage}>
      <div className={styles.recordToolbar}>
        <button type="button" className={styles.recordBack} onClick={() => navigate('/insights')}>
          <ProductIcon name="arrow-left" width={16} height={16} />
          Insights
        </button>
        <div className={styles.recordActions}>
          <button type="button" onClick={() => void loadRecord()} className={styles.recordAction}>
            <ProductIcon name="refresh" width={15} height={15} />
            Refresh
          </button>
          <button
            type="button"
            onClick={() => void handleCopyLink()}
            className={styles.recordActionPrimary}
          >
            <ProductIcon name={copyState === 'copied' ? 'check' : 'copy'} width={15} height={15} />
            {copyState === 'copied' ? 'Copied' : 'Copy link'}
          </button>
        </div>
      </div>

      {error ? (
        <div className={styles.error}>
          <span>{error}</span>
        </div>
      ) : null}

      {loading ? <ProductLoadingState label="Loading insight" compact /> : null}

      {!loading && !error && record ? (
        <>
          <header className={styles.recordHero}>
            <div className={styles.recordHeroCopy}>
              <span className={styles.recordEyebrow}>Request insight</span>
              <h1>{buildInsightsRecordTitle(record)}</h1>
              <p>One request, from routing decision to delivered response.</p>
            </div>
            {lifecycle ? (
              <span
                className={`${styles.recordStatus} ${
                  lifecycle.successful
                    ? styles.recordStatusSuccess
                    : lifecycle.errored
                      ? styles.recordStatusError
                      : styles.recordStatusNeutral
                }`}
              >
                {lifecycle.label}
              </span>
            ) : null}
          </header>

          <div className={styles.recordSections}>
            {sections.map((section, sectionIndex) => (
              <section
                key={`${section.title ?? 'details'}-${sectionIndex}`}
                className={styles.recordSection}
              >
                {section.title ? <h2>{section.title}</h2> : null}
                <div className={styles.recordFields}>
                  {section.fields.map((field, fieldIndex) => (
                    <div
                      key={`${field.label}-${fieldIndex}`}
                      className={`${styles.recordField} ${field.fullWidth ? styles.recordFieldWide : ''}`}
                    >
                      <span>{field.label}</span>
                      <div>{field.value}</div>
                    </div>
                  ))}
                </div>
              </section>
            ))}
          </div>
        </>
      ) : null}
    </main>
  )
}
