import React from 'react'
import { useReadonly } from '../contexts/ReadonlyContext'
import ProductIcon from './ProductIcon'
import styles from './ReadonlyBanner.module.css'

const ReadonlyBanner: React.FC = () => {
  const { isReadonly } = useReadonly()

  if (!isReadonly) {
    return null
  }

  return (
    <div className={styles.banner}>
      <ProductIcon className={styles.icon} name="shield" aria-hidden="true" />
      <span className={styles.text}>
        Dashboard is in read-only mode. Configuration editing is disabled.
      </span>
    </div>
  )
}

export default ReadonlyBanner
