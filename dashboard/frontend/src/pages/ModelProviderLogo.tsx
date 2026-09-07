import { useEffect, useState } from 'react'

import { isMonochromeModelProviderIcon, type ModelProviderPreset } from './modelProviderCatalog'
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
  const initialSource = provider?.icon || fallbackSource
  const [source, setSource] = useState(initialSource)
  const [failed, setFailed] = useState(false)

  useEffect(() => {
    setSource(initialSource)
    setFailed(false)
  }, [initialSource])

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
      {!failed ? (
        <img
          src={source}
          alt=""
          referrerPolicy="no-referrer"
          data-monochrome={isMonochromeModelProviderIcon(source)}
          onError={handleError}
        />
      ) : (
        <span>{provider?.monogram || 'v'}</span>
      )}
    </span>
  )
}
