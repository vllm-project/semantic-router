import React from 'react'
import clsx from 'clsx'
import styles from './GradientMesh.module.css'

interface GradientMeshProps {
  className?: string
  variant?: 'hero' | 'subtle'
}

export default function GradientMesh({
  className,
  variant = 'hero',
}: GradientMeshProps): JSX.Element {
  return (
    <div
      className={clsx(styles.mesh, variant === 'subtle' && styles.meshSubtle, className)}
      aria-hidden="true"
    >
      <div className={styles.orbViolet} />
      <div className={styles.orbBlue} />
      <div className={styles.orbCyan} />
      <div className={styles.orbIndigo} />
    </div>
  )
}
