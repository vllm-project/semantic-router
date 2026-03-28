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
  groups?: Record<string, string[]>
}

function buildKnowledgeMapURL(name: string): string {
  const encodedName = encodeURIComponent(name)
  const params = new URLSearchParams({
    metadataURL: withAuthQuery(`/api/router/config/kbs/${encodedName}/map/metadata`),
    dataURL: withAuthQuery(`/api/router/config/kbs/${encodedName}/map/data.ndjson`),
    title: name,
  })
  return withAuthQuery(`/embedded/wizmap/?${params.toString()}`)
}

export default function KnowledgeMapPage() {
  const { name = '' } = useParams<{ name: string }>()
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
      .then(() => {
        if (cancelled) {
          return
        }
      })
      .catch((nextError) => {
        if (!cancelled) {
          setError(nextError instanceof Error ? nextError.message : 'Failed to load knowledge map metadata')
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
      {error ? <div className={styles.error}>{error}</div> : null}

      <section className={styles.mapShell}>
        <Link to="/knowledge-bases/bases" className={styles.backLink}>
          Back to Bases
        </Link>

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
