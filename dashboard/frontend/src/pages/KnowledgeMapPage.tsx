import { useEffect, useMemo, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import { withAuthQuery } from '../utils/authFetch'
import styles from './KnowledgeMapPage.module.css'

interface KnowledgeMapMetadata {
  name: string
  description?: string
  projection: string
  model_type: string
  point_count: number
  label_count: number
  group_count: number
  label_names: string[]
  topic_label_hint?: string[]
}

function buildKnowledgeMapURL(name: string): string {
  const encodedName = encodeURIComponent(name)
  const params = new URLSearchParams({
    dataURL: withAuthQuery(`/api/router/config/kbs/${encodedName}/map/data.ndjson`),
    gridURL: withAuthQuery(`/api/router/config/kbs/${encodedName}/map/grid.json`),
    topicURL: withAuthQuery(`/api/router/config/kbs/${encodedName}/map/topic.json`),
    title: name,
  })
  return withAuthQuery(`/embedded/wizmap/?${params.toString()}`)
}

export default function KnowledgeMapPage() {
  const { name = '' } = useParams<{ name: string }>()
  const [metadata, setMetadata] = useState<KnowledgeMapMetadata | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [iframeReady, setIframeReady] = useState(false)

  useEffect(() => {
    let cancelled = false
    setLoading(true)
    setError(null)
    setIframeReady(false)

    fetch(`/api/router/config/kbs/${encodeURIComponent(name)}/map/metadata`)
      .then(async (response) => {
        if (!response.ok) {
          const message = await response.text()
          throw new Error(message || `HTTP ${response.status}: ${response.statusText}`)
        }
        return response.json() as Promise<KnowledgeMapMetadata>
      })
      .then((nextMetadata) => {
        if (!cancelled) {
          setMetadata(nextMetadata)
        }
      })
      .catch((nextError) => {
        if (!cancelled) {
          setError(nextError instanceof Error ? nextError.message : 'Failed to load knowledge map metadata')
          setMetadata(null)
        }
      })
      .finally(() => {
        if (!cancelled) {
          setLoading(false)
        }
      })

    return () => {
      cancelled = true
    }
  }, [name])

  const iframeURL = useMemo(() => buildKnowledgeMapURL(name), [name])

  return (
    <div className={styles.page}>
      <header className={styles.header}>
        <div className={styles.headerMain}>
          <div className={styles.eyebrow}>Knowledge</div>
          <div className={styles.titleRow}>
            <div className={styles.titleStack}>
              <h1 className={styles.title}>Knowledge Map</h1>
              <p className={styles.description}>
                Inspect one base at a time with the self-hosted WizMap viewer.
              </p>
            </div>
            <Link to="/knowledge-bases/bases" className={styles.backLink}>
              Back to Bases
            </Link>
          </div>
        </div>

        {metadata ? (
          <div className={styles.summaryBar}>
            <div className={styles.summaryPrimary}>
              <span className={styles.summaryEyebrow}>Base</span>
              <strong className={styles.summaryName}>{metadata.name}</strong>
              {metadata.description ? (
                <p className={styles.summaryDescription}>{metadata.description}</p>
              ) : null}
            </div>
            <dl className={styles.summaryStats}>
              <div className={styles.summaryStat}>
                <dt>Projection</dt>
                <dd>{metadata.projection}</dd>
              </div>
              <div className={styles.summaryStat}>
                <dt>Model</dt>
                <dd>{metadata.model_type}</dd>
              </div>
              <div className={styles.summaryStat}>
                <dt>Points</dt>
                <dd>{metadata.point_count}</dd>
              </div>
              <div className={styles.summaryStat}>
                <dt>Labels</dt>
                <dd>{metadata.label_count}</dd>
              </div>
            </dl>
          </div>
        ) : null}
      </header>

      {error ? <div className={styles.error}>{error}</div> : null}

      <section className={styles.mapShell}>
        {(loading || !iframeReady) && !error ? (
          <div className={styles.loadingPanel}>
            <div className={styles.spinner} />
            <p>{loading ? 'Loading knowledge map metadata...' : 'Launching WizMap...'}</p>
          </div>
        ) : null}
        {!error ? (
          <iframe
            key={iframeURL}
            src={iframeURL}
            title={`Knowledge map for ${name}`}
            className={styles.iframe}
            onLoad={() => setIframeReady(true)}
          />
        ) : null}
      </section>
    </div>
  )
}
