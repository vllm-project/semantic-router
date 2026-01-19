import React from 'react'
import { useReadonly } from '../contexts/ReadonlyContext'
import styles from './ReadonlyBanner.module.css'

const ReadonlyBanner: React.FC = () => {
  const { isReadonly } = useReadonly()

  if (!isReadonly) {
    return null
  }

  return (
    <div className={styles.banner}>
      <span className={styles.icon}>🔒</span>
      <span className={styles.text}>
        Dashboard is in read-only mode. Configuration editing is disabled.
      </span>
    </div>
  )
}

export default ReadonlyBanner
