import { useState } from 'react'
import { benchApi } from './api'
import { number } from './model'
import styles from './SrBench.module.css'

export default function RunArtifacts({ id, canRun }: { id: string; canRun: boolean }) {
  const [pending, setPending] = useState(false)
  const [artifact, setArtifact] = useState<Record<string, unknown> | null>(null)
  const [error, setError] = useState('')
  async function create(kind: 'regrade' | 'export') {
    setPending(true)
    setError('')
    setArtifact(null)
    try {
      setArtifact(await (kind === 'regrade' ? benchApi.regrade(id) : benchApi.exportMatrix(id)))
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Could not prepare this artifact.')
    } finally {
      setPending(false)
    }
  }
  function download() {
    if (!artifact) return
    const url = URL.createObjectURL(
      new Blob([JSON.stringify(artifact, null, 2)], { type: 'application/json' }),
    )
    const link = document.createElement('a')
    link.href = url
    link.download = `${id}-${artifact.kind}.json`
    link.click()
    setTimeout(() => URL.revokeObjectURL(url), 1000)
  }
  return (
    <section>
      <h3>Reuse saved evidence</h3>
      <p>
        Regrade deterministic multiple-choice or grid answers without model calls. Export a
        development-only model matrix for router training; holdout and unknown splits are rejected.
      </p>
      <div className={styles.actions}>
        <button disabled={!canRun || pending} onClick={() => void create('regrade')}>
          Regrade saved answers
        </button>
        <button disabled={!canRun || pending} onClick={() => void create('export')}>
          Export training matrix
        </button>
      </div>
      {pending && <p role="status">Preparing saved evidence…</p>}
      {error && (
        <p className={styles.error} role="alert">
          {error}
        </p>
      )}
      {artifact && (
        <div className={styles.notice}>
          <strong>
            {artifact.kind === 'offline-regrade'
              ? 'Offline regrade ready'
              : 'Development matrix ready'}
          </strong>
          {artifact.kind === 'offline-regrade' && (
            <p>
              {number(artifact.changed_count)} changed grades · {number(artifact.model_requests)}{' '}
              model requests
            </p>
          )}
          <button onClick={download}>Download artifact JSON</button>
          <details>
            <summary>Inspect artifact</summary>
            <pre>{JSON.stringify(artifact, null, 2)}</pre>
          </details>
        </div>
      )}
    </section>
  )
}
