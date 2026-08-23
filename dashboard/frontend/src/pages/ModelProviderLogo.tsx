import { useEffect, useState, type CSSProperties } from 'react'

import type { ProviderCatalogIcon } from '../utils/providerCatalogApi'
import { getProviderIconAsset } from './modelProviderLogoSupport'
import styles from './ModelProviderLogo.module.css'

interface ModelProviderLogoProps {
  icon?: ProviderCatalogIcon
  name?: string
  monogram?: string
  accent?: string
  size?: 'small' | 'medium' | 'large'
  className?: string
}

export default function ModelProviderLogo({
  icon,
  name = 'Model provider',
  monogram,
  accent = '#8b8b92',
  size = 'medium',
  className = '',
}: ModelProviderLogoProps) {
  const [imageFailed, setImageFailed] = useState(false)
  const imageSource = getProviderIconAsset(icon)
  useEffect(() => setImageFailed(false), [icon?.source, icon?.value, icon?.color])
  const style = { '--provider-accent': accent } as CSSProperties
  const fallback =
    monogram?.trim() ||
    name
      .split(/\s+/)
      .map((part) => part[0])
      .join('')
      .slice(0, 2)
      .toUpperCase() ||
    'M'
  return (
    <span
      className={`${styles.logo} ${styles[size]} ${className}`}
      style={style}
      title={name}
      aria-label={`${name} logo`}
    >
      {imageSource && !imageFailed ? (
        <img
          src={imageSource}
          alt=""
          referrerPolicy="no-referrer"
          className={icon?.color ? styles.colorIcon : styles.monoIcon}
          onError={() => setImageFailed(true)}
        />
      ) : (
        <span>{fallback}</span>
      )}
    </span>
  )
}
