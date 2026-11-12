import React from 'react'
import clsx from 'clsx'
import BlockBand, { type BandTone } from './BlockBand'
import MosaicRule from './MosaicRule'
import styles from './Ground.module.css'

type GroundProps = {
  children: React.ReactNode
  variant: 'paper' | 'blueprint' | 'band'
  tone?: BandTone
  rule?: boolean
  seed?: number
}

/* The ground a homepage section sits on, as on System Duo: plain paper, a dotted blueprint band, or a band of blocks;
 * optionally a mosaic rule above the section's head. None of them draws an edge: each fades in from the paper and
 * back out into it, so neighbouring sections never meet at a line. */
export default function Ground({ children, variant, tone = 'blue', rule = false, seed = 1 }: GroundProps): JSX.Element {
  return (
    <div className={clsx(styles.ground, styles[variant], rule && styles.ruled)}>
      {variant === 'band' && <BlockBand tone={tone} seed={seed} />}
      {rule && <MosaicRule className={styles.groundRule} seed={seed} />}
      {children}
    </div>
  )
}
