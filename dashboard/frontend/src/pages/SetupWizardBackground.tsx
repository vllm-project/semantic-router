import { Suspense, lazy, useEffect, useState } from 'react'
import {
  DASHBOARD_COLOR_BENDS_MOTION,
  DASHBOARD_MOTION_COLORS,
} from '../components/dashboardMotionTheme'
import styles from './SetupWizardPage.module.css'

const ColorBends = lazy(() => import('../components/ColorBends'))

function readReducedMotionPreference() {
  return (
    typeof window === 'undefined' ||
    (window.matchMedia?.('(prefers-reduced-motion: reduce)').matches ?? false)
  )
}

export default function SetupWizardBackground() {
  const [reducedMotion, setReducedMotion] = useState(readReducedMotionPreference)

  useEffect(() => {
    const motionQuery = window.matchMedia?.('(prefers-reduced-motion: reduce)')
    if (!motionQuery) return

    const handleMotionChange = (event: MediaQueryListEvent) => {
      setReducedMotion(event.matches)
    }
    motionQuery.addEventListener?.('change', handleMotionChange)
    return () => motionQuery.removeEventListener?.('change', handleMotionChange)
  }, [])

  return (
    <div
      className={styles.backgroundEffect}
      data-testid="setup-motion-background"
      data-motion={reducedMotion ? 'reduced' : 'animated'}
      aria-hidden="true"
    >
      {reducedMotion ? null : (
        <Suspense fallback={<div className={styles.backgroundFallback} />}>
          <ColorBends
            colors={DASHBOARD_MOTION_COLORS}
            {...DASHBOARD_COLOR_BENDS_MOTION}
            transparent
          />
        </Suspense>
      )}
    </div>
  )
}
