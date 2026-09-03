import type { ReactNode } from 'react'

import ColorBends from '../components/ColorBends'
import {
  DASHBOARD_COLOR_BENDS_MOTION,
  DASHBOARD_MOTION_COLORS,
} from '../components/dashboardMotionTheme'
import styles from './AuthExperienceShell.module.css'

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
        <ColorBends
          colors={DASHBOARD_MOTION_COLORS}
          {...DASHBOARD_COLOR_BENDS_MOTION}
          transparent
        />
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
