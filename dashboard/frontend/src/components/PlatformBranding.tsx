import { useEffect, useState } from 'react'
import { useReadonly } from '../contexts/ReadonlyContext'
import styles from './PlatformBranding.module.css'

interface PlatformBrandingProps {
  variant?: 'default' | 'compact' | 'inline'
  className?: string
}

// Preload cache to track which images have been loaded
const preloadedImages = new Set<string>()

// Preload an image and cache it
const preloadImage = (src: string): Promise<void> => {
  if (preloadedImages.has(src)) {
    return Promise.resolve()
  }

  return new Promise((resolve) => {
    const img = new Image()
    img.onload = () => {
      preloadedImages.add(src)
      resolve()
    }
    img.onerror = () => resolve() // Resolve anyway to not block
    img.src = src
  })
}

const PlatformBranding = ({ variant = 'default', className = '' }: PlatformBrandingProps) => {
  const { platform } = useReadonly()
  const [isImageLoaded, setIsImageLoaded] = useState(false)

  const isAmd = platform?.toLowerCase() === 'amd'
  const imageSrc = '/amd.png'

  // Preload image when platform is AMD
  useEffect(() => {
    if (isAmd) {
      if (preloadedImages.has(imageSrc)) {
        setIsImageLoaded(true)
      } else {
        preloadImage(imageSrc).then(() => setIsImageLoaded(true))
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

// Export preload function for early loading
export const preloadPlatformAssets = (platform?: string) => {
  if (platform?.toLowerCase() === 'amd') {
    preloadImage('/amd.png')
  }
}

export default PlatformBranding

