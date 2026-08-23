import { useState } from 'react'

import { agentManagementApi } from '../utils/agentManagementApi'
import type { AgentArtifact, AgentArtifactContent } from '../generated/managementApiContract'
import ProductIcon from './ProductIcon'
import styles from './AgentPlayground.module.css'

const MAX_INLINE_TEXT_BYTES = 512 * 1024
const MAX_INLINE_IMAGE_BYTES = 4 * 1024 * 1024

interface AgentArtifactResultProps {
  artifactId: string
  canLoadOriginal: boolean
}

interface DecodedArtifact {
  bytes: number
  imageUrl?: string
  text?: string
}

function isTextMediaType(mediaType: string): boolean {
  return (
    mediaType.startsWith('text/') || /(?:json|yaml|xml|javascript|markdown)(?:;|$)/i.test(mediaType)
  )
}

function decodeBase64(content: AgentArtifactContent): DecodedArtifact {
  const binary = window.atob(content.content)
  const bytes = Uint8Array.from(binary, (character) => character.charCodeAt(0))
  if (
    /^image\/(?:png|jpeg|gif|webp|avif)(?:;|$)/i.test(content.mediaType) &&
    bytes.byteLength <= MAX_INLINE_IMAGE_BYTES
  ) {
    return {
      bytes: bytes.byteLength,
      imageUrl: `data:${content.mediaType};base64,${content.content}`,
    }
  }
  if (isTextMediaType(content.mediaType) && bytes.byteLength <= MAX_INLINE_TEXT_BYTES) {
    return {
      bytes: bytes.byteLength,
      text: new TextDecoder('utf-8', { fatal: true }).decode(bytes),
    }
  }
  return { bytes: bytes.byteLength }
}

function previewText(preview: AgentArtifact['safePreview']): string {
  try {
    return JSON.stringify(preview, null, 2)
  } catch {
    return 'Preview unavailable.'
  }
}

export default function AgentArtifactResult({
  artifactId,
  canLoadOriginal,
}: AgentArtifactResultProps) {
  const [artifact, setArtifact] = useState<AgentArtifact | null>(null)
  const [original, setOriginal] = useState<DecodedArtifact | null>(null)
  const [expanded, setExpanded] = useState(false)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const openPreview = async () => {
    if (artifact) {
      setExpanded((current) => !current)
      return
    }
    setLoading(true)
    setError(null)
    try {
      setArtifact(await agentManagementApi.getArtifact(artifactId))
      setExpanded(true)
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'The result is unavailable.')
    } finally {
      setLoading(false)
    }
  }

  const loadOriginal = async () => {
    if (original || loading) return
    setLoading(true)
    setError(null)
    try {
      const content = await agentManagementApi.getArtifactContent(artifactId)
      if (artifact && content.digest !== artifact.digest) {
        throw new Error('The result changed. Refresh this conversation.')
      }
      setOriginal(decodeBase64(content))
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'The original result is unavailable.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <section className={styles.artifactResult} aria-live="polite">
      <button type="button" onClick={() => void openPreview()} disabled={loading}>
        <ProductIcon name="eye" />
        {loading && !artifact ? 'Loading…' : artifact && expanded ? 'Hide result' : 'View result'}
      </button>
      {error ? (
        <p className={styles.toolError} role="alert">
          {error}
        </p>
      ) : null}
      {artifact && expanded ? (
        <div className={styles.artifactPanel}>
          <div className={styles.artifactMeta}>
            <span>{artifact.kind}</span>
            <span>{artifact.mediaType}</span>
          </div>
          <pre>{previewText(artifact.safePreview)}</pre>
          {canLoadOriginal && !original ? (
            <button type="button" onClick={() => void loadOriginal()} disabled={loading}>
              {loading ? 'Loading…' : 'Load original'}
            </button>
          ) : null}
          {original?.text !== undefined ? <pre>{original.text}</pre> : null}
          {original?.imageUrl ? <img src={original.imageUrl} alt="Artifact result" /> : null}
          {original && original.text === undefined && !original.imageUrl ? (
            <small>{original.bytes.toLocaleString('en-US')} bytes loaded.</small>
          ) : null}
        </div>
      ) : null}
    </section>
  )
}
