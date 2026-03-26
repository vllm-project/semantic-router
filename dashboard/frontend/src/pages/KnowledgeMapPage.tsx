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
    dataURL: `/api/router/config/kbs/${encodedName}/map/data.ndjson`,
    gridURL: `/api/router/config/kbs/${encodedName}/map/grid.json`,
    topicURL: `/api/router/config/kbs/${encodedName}/map/topic.json`,
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
            <h1 className={styles.title}>Knowledge Map</h1>
            <Link to="/knowledge-bases/bases" className={styles.backLink}>
              Back to Bases
            </Link>
          </div>
          <p className={styles.description}>
            Inspect one knowledge base at a time with the self-hosted WizMap viewer. The current map is built from exemplar embeddings and grouped by label.
          </p>
        </div>

        {metadata ? (
          <div className={styles.metaRail}>
            <div className={styles.metaCard}>
              <span className={styles.metaLabel}>Base</span>
              <strong className={styles.metaValue}>{metadata.name}</strong>
              <span className={styles.metaHint}>{metadata.description || 'No description provided.'}</span>
            </div>
            <div className={styles.metaGrid}>
              <div className={styles.metaTile}>
                <span className={styles.metaLabel}>Projection</span>
                <strong className={styles.metaValue}>{metadata.projection}</strong>
              </div>
              <div className={styles.metaTile}>
                <span className={styles.metaLabel}>Embedding Model</span>
                <strong className={styles.metaValue}>{metadata.model_type}</strong>
              </div>
              <div className={styles.metaTile}>
                <span className={styles.metaLabel}>Points</span>
                <strong className={styles.metaValue}>{metadata.point_count}</strong>
              </div>
              <div className={styles.metaTile}>
                <span className={styles.metaLabel}>Labels</span>
                <strong className={styles.metaValue}>{metadata.label_count}</strong>
              </div>
            </div>
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
