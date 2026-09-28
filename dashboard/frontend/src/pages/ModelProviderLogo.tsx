import { useEffect, useState } from 'react'

import type { ModelProviderPreset } from './modelProviderCatalog'
import styles from './ModelProviderLogo.module.css'

interface ModelProviderLogoProps {
  provider?: ModelProviderPreset
  size?: 'small' | 'medium' | 'large'
  fallbackSource?: string
}

export default function ModelProviderLogo({
  provider,
  size = 'medium',
  fallbackSource = '/vllm.png',
}: ModelProviderLogoProps) {
  // A missing catalog icon is intentional: render the provider's monogram.
  // The vLLM image is only the anonymous-model fallback and must never leak
  // into unrelated provider cards.
  const initialSource = provider ? provider.icon : fallbackSource
  const usesMonogram = Boolean(provider && !provider.icon)
  const [source, setSource] = useState(initialSource)
  const [failed, setFailed] = useState(usesMonogram)

  useEffect(() => {
    setSource(initialSource)
    setFailed(usesMonogram)
  }, [initialSource, usesMonogram])

  const handleError = () => {
    if (provider) {
      setFailed(true)
      return
    }
    if (source !== fallbackSource) {
      setSource(fallbackSource)
      return
    }
    setFailed(true)
  }

  return (
    <span
      className={`${styles.logo} ${styles[size]}`}
      aria-label={`${provider?.name ?? 'vLLM'} logo`}
      title={provider?.name ?? 'vLLM'}
    >
      {source && !failed ? (
        <img
          src={source}
          alt=""
          referrerPolicy="no-referrer"
          data-monochrome={Boolean(provider?.monochrome)}
          onError={handleError}
        />
      ) : (
        <span>{provider?.monogram || 'v'}</span>
      )}
    </span>
  )
}
