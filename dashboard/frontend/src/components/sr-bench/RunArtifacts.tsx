import { useState } from 'react'
import { Download } from 'lucide-react'
import { Button } from '../../ui'
import { benchApi } from './api'
import { number } from './model'
import { ActionRow, BenchBlock, Disclosure, Notice } from './benchUi'
import { benchText, codeBlockClass } from './benchStyles'

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
    <BenchBlock className="mt-5">
      <h3 className={benchText.h3}>Reuse saved evidence</h3>
      <p className={`${benchText.muted} mt-1 max-w-3xl`}>
        Regrade deterministic multiple-choice or grid answers without model calls. Export a
        development-only model matrix for router training; holdout and unknown splits are rejected.
      </p>
      <ActionRow className="mt-3">
        <Button
          variant="outline"
          size="sm"
          disabled={!canRun || pending}
          onClick={() => void create('regrade')}
        >
          Regrade saved answers
        </Button>
        <Button
          variant="outline"
          size="sm"
          disabled={!canRun || pending}
          onClick={() => void create('export')}
        >
          Export training matrix
        </Button>
      </ActionRow>
      {pending && (
        <p role="status" className={`${benchText.muted} mt-2`}>
          Preparing saved evidence…
        </p>
      )}
      {error && (
        <Notice tone="danger" as="p" role="alert">
          {error}
        </Notice>
      )}
      {artifact && (
        <Notice tone="success">
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
          <div>
            <Button variant="outline" size="xs" onClick={download}>
              <Download aria-hidden="true" />
              Download artifact JSON
            </Button>
          </div>
          <Disclosure summary="Inspect artifact" bare className="my-0">
            <pre className={codeBlockClass}>{JSON.stringify(artifact, null, 2)}</pre>
          </Disclosure>
        </Notice>
      )}
    </BenchBlock>
  )
}
