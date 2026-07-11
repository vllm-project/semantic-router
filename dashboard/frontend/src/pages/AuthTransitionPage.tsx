import React, { useEffect, useState } from 'react'
import { Navigate, useNavigate, useSearchParams } from 'react-router-dom'
import { useAuth } from '../contexts/AuthContext'
import { useSetup } from '../contexts/SetupContext'
import AuthTransitionScene from './AuthTransitionScene'
import {
  AUTH_TRANSITION_MIN_DURATION_MS,
  sanitizeAuthTransitionTarget,
} from './authTransitionSupport'
import styles from './AuthTransitionPage.module.css'

type Milestone = {
  key: string
  label: string
  detail: string
  revealAt: number
}

const PROGRESS_SEGMENTS = [
  { duration: 400, from: 0, to: 24 },
  { duration: 500, from: 24, to: 54 },
  { duration: 600, from: 54, to: 84 },
  { duration: 800, from: 84, to: 100 },
]

const MILESTONES: Milestone[] = [
  {
    key: 'session',
    label: 'Session verified',
    detail: 'Secure access confirmed',
    revealAt: 8,
  },
  {
    key: 'signals',
    label: 'Signals synchronized',
    detail: 'Routing context loaded',
    revealAt: 34,
  },
  {
    key: 'decisions',
    label: 'Decisions composed',
    detail: 'Policy graph prepared',
    revealAt: 62,
  },
  {
    key: 'route',
    label: 'Route ready',
    detail: 'Model path prepared',
    revealAt: 86,
  },
]

function easeOutCubic(value: number): number {
  return 1 - (1 - value) ** 3
}

function usePrefersReducedMotion(): boolean {
  const [prefersReducedMotion, setPrefersReducedMotion] = useState(() => {
    return Boolean(
      typeof window !== 'undefined' &&
        window.matchMedia?.('(prefers-reduced-motion: reduce)').matches,
    )
  })

  useEffect(() => {
    const mediaQuery = window.matchMedia?.('(prefers-reduced-motion: reduce)')
    if (!mediaQuery) return

    const handleChange = (event: MediaQueryListEvent) => setPrefersReducedMotion(event.matches)
    mediaQuery.addEventListener('change', handleChange)
    return () => mediaQuery.removeEventListener('change', handleChange)
  }, [])

  return prefersReducedMotion
}

function getTransitionProgress(elapsedMs: number): number {
  let consumedDuration = 0

  for (const segment of PROGRESS_SEGMENTS) {
    const segmentEnd = consumedDuration + segment.duration
    if (elapsedMs <= segmentEnd) {
      const localProgress = (elapsedMs - consumedDuration) / segment.duration
      const easedProgress = easeOutCubic(Math.max(0, Math.min(localProgress, 1)))
      return segment.from + (segment.to - segment.from) * easedProgress
    }
    consumedDuration = segmentEnd
  }

  return 100
}

function getActiveMilestoneIndex(progress: number): number {
  return MILESTONES.reduce((activeIndex, milestone, index) => {
    return progress >= milestone.revealAt ? index : activeIndex
  }, 0)
}

const AuthTransitionPage: React.FC = () => {
  const navigate = useNavigate()
  const [searchParams] = useSearchParams()
  const { isAuthenticated, isLoading } = useAuth()
  const { setupState } = useSetup()
  const prefersReducedMotion = usePrefersReducedMotion()
  const [progress, setProgress] = useState(0)
  const [animationComplete, setAnimationComplete] = useState(false)

  const fallbackTarget = setupState?.setupMode ? '/setup' : '/dashboard'
  const target = sanitizeAuthTransitionTarget(searchParams.get('to'), fallbackTarget)
  const activeMilestoneIndex = getActiveMilestoneIndex(progress)
  const activeMilestone = MILESTONES[activeMilestoneIndex]

  useEffect(() => {
    if (prefersReducedMotion) {
      setProgress(100)
      setAnimationComplete(true)
      return
    }

    setAnimationComplete(false)
    setProgress(0)
    const startTime = performance.now()
    let isComplete = false
    const timers: { progress?: number; completion?: number } = {}

    const completeTransition = () => {
      if (isComplete) return

      isComplete = true
      if (timers.progress !== undefined) window.clearInterval(timers.progress)
      if (timers.completion !== undefined) window.clearTimeout(timers.completion)
      setProgress(100)
      setAnimationComplete(true)
    }

    const updateProgress = () => {
      const elapsed = performance.now() - startTime
      if (elapsed >= AUTH_TRANSITION_MIN_DURATION_MS) {
        completeTransition()
        return
      }
      setProgress(getTransitionProgress(elapsed))
    }

    updateProgress()
    timers.progress = window.setInterval(updateProgress, 50)
    timers.completion = window.setTimeout(completeTransition, AUTH_TRANSITION_MIN_DURATION_MS)

    const handleVisibilityChange = () => updateProgress()
    document.addEventListener('visibilitychange', handleVisibilityChange)

    return () => {
      if (timers.progress !== undefined) window.clearInterval(timers.progress)
      if (timers.completion !== undefined) window.clearTimeout(timers.completion)
      document.removeEventListener('visibilitychange', handleVisibilityChange)
    }
  }, [prefersReducedMotion])

  useEffect(() => {
    if (animationComplete && isAuthenticated && !isLoading) {
      navigate(target, { replace: true })
    }
  }, [animationComplete, isAuthenticated, isLoading, navigate, target])

  if (!isAuthenticated && !isLoading) {
    return <Navigate to="/login" replace state={{ from: target }} />
  }

  return (
    <main className={styles.page} data-testid="auth-transition" aria-busy={!animationComplete}>
      <AuthTransitionScene progress={progress} reducedMotion={prefersReducedMotion} />

      <header className={styles.header}>
        <span className={styles.brand}>vLLM Semantic Router</span>
        <span className={styles.handoff}>
          <span className={styles.statusBeacon} aria-hidden="true" />
          Secure handoff
        </span>
      </header>

      <section className={styles.copy}>
        <span className={styles.eyebrow}>Access confirmed</span>
        <h1 className={styles.title}>Entering control plane</h1>
        <p className={styles.activeStage} data-testid="auth-transition-stage">
          <span className={styles.activeStageIndex}>
            {String(activeMilestoneIndex + 1).padStart(2, '0')}
          </span>
          <span>
            <strong>{activeMilestone.label}</strong>
            <span>{activeMilestone.detail}</span>
          </span>
        </p>
      </section>

      <span className={styles.srOnly} aria-live="polite">
        {activeMilestone.label}. {activeMilestone.detail}.
      </span>

      <footer className={styles.footer}>
        <ol className={styles.phaseRail} aria-hidden="true">
          {MILESTONES.map((milestone, index) => {
            const isReached = progress >= milestone.revealAt
            const isActive = index === activeMilestoneIndex
            return (
              <li
                key={milestone.key}
                className={`${styles.phase} ${isReached ? styles.phaseReached : ''} ${isActive ? styles.phaseActive : ''}`}
              >
                <span className={styles.phaseDot} />
                <span className={styles.phaseLabel}>{milestone.label}</span>
              </li>
            )
          })}
        </ol>

        <div
          className={styles.progressTrack}
          role="progressbar"
          aria-label="Opening workspace"
          aria-valuemin={0}
          aria-valuemax={100}
          aria-valuenow={Math.round(progress)}
        >
          <div className={styles.progressBar} style={{ transform: `scaleX(${progress / 100})` }} />
        </div>
      </footer>
    </main>
  )
}

export default AuthTransitionPage
