import { useEffect, useState } from 'react'
import { useReadonly } from '../contexts/ReadonlyContext'
import { isImagePreloaded, preloadImageSrc } from '../utils/platformAssets'
import styles from './PlatformBranding.module.css'

interface PlatformBrandingProps {
  variant?: 'default' | 'compact' | 'inline'
  className?: string
}

const PlatformBranding = ({ variant = 'default', className = '' }: PlatformBrandingProps) => {
  const { platform } = useReadonly()
  const [isImageLoaded, setIsImageLoaded] = useState(false)

  const isAmd = platform?.toLowerCase() === 'amd'
  const imageSrc = '/amd.png'

  // Preload image when platform is AMD
  useEffect(() => {
    if (isAmd) {
      if (isImagePreloaded(imageSrc)) {
        setIsImageLoaded(true)
      } else {
        preloadImageSrc(imageSrc).then(() => setIsImageLoaded(true))
      }
    }
  }, [isAmd])

  // Only show branding if platform is AMD and image is loaded
  if (!isAmd || !isImageLoaded) {
    return null
  }

  return (
    <div className={`${styles.container} ${styles[variant]} ${className}`}>
      <img
        src={imageSrc}
        alt="AMD"
        className={styles.logo}
      />
      <span className={styles.text}>Powered by AMD GPU</span>
    </div>
  )
}

export default PlatformBranding

