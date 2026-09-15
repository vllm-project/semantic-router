import { Link } from 'react-router-dom'
import styles from './BrandLockup.module.css'

interface BrandLockupProps {
  className?: string
  to?: string
}

export default function BrandLockup({ className = '', to = '/' }: BrandLockupProps) {
  return (
    <Link
      className={`${styles.brand} ${className}`.trim()}
      to={to}
      aria-label="vLLM Semantic Router home"
    >
      <img className={styles.logo} src="/vllm-sr-logo.white.png" alt="" aria-hidden="true" />
    </Link>
  )
}
