import { useReadonly } from '../contexts/ReadonlyContext'
import styles from './PlatformBranding.module.css'

interface PlatformBrandingProps {
  variant?: 'default' | 'compact' | 'inline'
  className?: string
}

const PlatformBranding = ({ variant = 'default', className = '' }: PlatformBrandingProps) => {
  const { platform } = useReadonly()

  // Only show branding if platform is set (e.g., 'amd')
  if (!platform || platform.toLowerCase() !== 'amd') {
    return null
  }

  return (
    <div className={`${styles.container} ${styles[variant]} ${className}`}>
      <img 
        src="/amd.png" 
        alt="AMD" 
        className={styles.logo}
      />
      <span className={styles.text}>Powered by AMD GPU</span>
    </div>
  )
}

export default PlatformBranding

