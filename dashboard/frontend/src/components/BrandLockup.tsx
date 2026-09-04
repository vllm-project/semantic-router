import { Link } from 'react-router-dom'
import styles from './BrandLockup.module.css'

interface BrandLockupProps {
  className?: string
}

export default function BrandLockup({ className = '' }: BrandLockupProps) {
  return (
    <Link
      className={`${styles.brand} ${className}`.trim()}
      to="/"
      aria-label="vLLM Semantic Router home"
    >
      <img className={styles.logo} src="/vllm-sr-logo.white.png" alt="" aria-hidden="true" />
    </Link>
  )
}
