import { useEffect, useState, type CSSProperties } from 'react'

import { getModelProvider } from './modelProviderCatalog'
import {
  getModelProviderLogoSource,
  providerDirectIcons,
  providerLobeIcons,
} from './modelProviderLogoSupport'
import styles from './ModelProviderLogo.module.css'

interface ModelProviderLogoProps {
  provider?: string
  size?: 'small' | 'medium' | 'large'
  className?: string
}

export default function ModelProviderLogo({
  provider,
  size = 'medium',
  className = '',
}: ModelProviderLogoProps) {
  const definition = getModelProvider(provider)
  const [imageFailed, setImageFailed] = useState(false)
  const icon = providerLobeIcons[definition.id]
  const imageSource = getModelProviderLogoSource(definition.id)
  useEffect(() => setImageFailed(false), [definition.id])
  const style = { '--provider-accent': definition.accent } as CSSProperties
  return (
    <span
      className={`${styles.logo} ${styles[size]} ${className}`}
      style={style}
      title={definition.name}
      aria-label={`${definition.name} logo`}
    >
      {imageSource && !imageFailed ? (
        <img
          src={imageSource}
          alt=""
          className={
            icon?.color || providerDirectIcons[definition.id] ? styles.colorIcon : styles.monoIcon
          }
          onError={() => setImageFailed(true)}
        />
      ) : (
        <span>{definition.shortName}</span>
      )}
    </span>
  )
}
