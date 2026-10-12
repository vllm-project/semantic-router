import { lazy, Suspense, type ReactNode } from 'react'

import {
  DASHBOARD_COLOR_BENDS_MOTION,
  DASHBOARD_MOTION_COLORS,
} from '../components/dashboardMotionTheme'
import styles from './AuthExperienceShell.module.css'

const ColorBends = lazy(() => import('../components/ColorBends'))

interface AuthExperienceShellProps {
  story: ReactNode
  children: ReactNode
}

export default function AuthExperienceShell({ story, children }: AuthExperienceShellProps) {
  return (
    <div className={styles.container}>
      <div
        className={styles.backgroundEffect}
        data-testid="login-motion-background"
        aria-hidden="true"
      >
        <Suspense fallback={null}>
          <ColorBends
            colors={DASHBOARD_MOTION_COLORS}
            {...DASHBOARD_COLOR_BENDS_MOTION}
            transparent
          />
        </Suspense>
      </div>
      <main className={styles.mainContent}>
        <div className={styles.shell}>
          <section className={styles.storyPanel}>{story}</section>
          {children}
        </div>
      </main>
    </div>
  )
}
