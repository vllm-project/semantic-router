import React, { useEffect, useId, useState } from 'react'
import { useLocation, useNavigate } from 'react-router-dom'
import {
  clearOnboardingStep,
  getOnboardingStep,
  getOnboardingStatus,
  setOnboardingStep,
  setOnboardingStatus,
  type OnboardingStatus,
} from '../utils/onboarding'
import { preloadDashboardRoute } from '../app/routeLoaders'
import useAccessibleDialog from '../hooks/useAccessibleDialog'
import styles from './OnboardingGuide.module.css'

interface GuideStep {
  id: string
  pageLabel: string
  title: string
  description: string
  highlights: string[]
  route: string
  actionLabel: string
}

const GUIDE_STEPS: GuideStep[] = [
  {
    id: 'models',
    pageLabel: 'Models',
    title: 'Start with the model inventory',
    description:
      'This page defines the models and endpoints the router can actually use before any routing logic becomes meaningful.',
    highlights: [
      'Register local or hosted model providers',
      'Choose the default model used by fallback routes',
      'Tune endpoint weights and credentials before touching routing',
    ],
    route: '/config/models',
    actionLabel: 'Open Models',
  },
  {
    id: 'routing',
    pageLabel: 'Decisions',
    title: 'Turn signals into routing behavior',
    description:
      'This is where request signals and explicit preferences become executable model paths.',
    highlights: [
      'Turn reusable signals and preference policy into executable model paths',
      'Choose when to select, cascade, or coordinate models',
      'Review the complete path before promoting changes',
    ],
    route: '/config/decisions',
    actionLabel: 'Open Decisions',
  },
  {
    id: 'playground',
    pageLabel: 'Playground',
    title: 'Test the active router end to end',
    description:
      'Use Playground as the shortest loop for checking whether the router is behaving the way you expect after setup.',
    highlights: [
      'Send prompts through the live routing pipeline',
      'Check whether the active routing graph behaves as expected',
      'Iterate here before changing real traffic',
    ],
    route: '/playground',
    actionLabel: 'Open Playground',
  },
  {
    id: 'dsl',
    pageLabel: 'DSL Builder',
    title: 'Author router behavior directly in DSL',
    description: 'Use Builder when the manager UI is no longer expressive enough.',
    highlights: [
      'Open the Guide drawer for DSL snippets',
      'Author model cards, signals, routes, and plugins',
      'Compile and deploy deeper routing changes',
    ],
    route: '/builder',
    actionLabel: 'Open DSL Builder',
  },
  {
    id: 'clawos',
    pageLabel: 'ClawOS',
    title: 'Orchestrate multi-claw worker systems',
    description: 'Use ClawOS when one router needs multi-agent orchestration.',
    highlights: [
      'Create teams with one leader and workers',
      'Connect workers to routed models and memory',
      'Inspect live agents, teams, and runtime health',
    ],
    route: '/clawos',
    actionLabel: 'Open ClawOS',
  },
]

const OnboardingGuide: React.FC = () => {
  const navigate = useNavigate()
  const location = useLocation()
  const [isOpen, setIsOpen] = useState(false)
  const [stepIndex, setStepIndex] = useState(0)
  const [isReady, setIsReady] = useState(false)
  const [status, setStatus] = useState<OnboardingStatus>('idle')
  const titleId = useId()
  const descriptionId = useId()

  useEffect(() => {
    const storedStatus = getOnboardingStatus()
    setStatus(storedStatus)
    setStepIndex(getOnboardingStep(GUIDE_STEPS.length))
    setIsOpen(storedStatus === 'pending')
    setIsReady(true)
  }, [])

  const handlePause = () => {
    setOnboardingStep(stepIndex)
    setOnboardingStatus('dismissed')
    setStatus('dismissed')
    setIsOpen(false)
  }

  const dialogRef = useAccessibleDialog<HTMLDivElement>({
    isOpen,
    onClose: handlePause,
  })

  if (!isReady || location.pathname === '/') {
    return null
  }

  const step = GUIDE_STEPS[stepIndex]
  const isOnTargetRoute = location.pathname === step.route

  const handleOpenGuide = () => {
    const resumeStep = status === 'idle' ? 0 : getOnboardingStep(GUIDE_STEPS.length)
    setOnboardingStep(resumeStep)
    setOnboardingStatus('pending')
    setStatus('pending')
    setStepIndex(resumeStep)
    setIsOpen(true)
  }

  const handleNext = () => {
    if (stepIndex === GUIDE_STEPS.length - 1) {
      setOnboardingStatus('completed')
      clearOnboardingStep()
      setStatus('completed')
      setIsOpen(false)
      return
    }

    setStepIndex((current) => {
      const next = current + 1
      setOnboardingStep(next)
      return next
    })
  }

  const handleBack = () => {
    setStepIndex((current) => {
      const next = current === 0 ? current : current - 1
      setOnboardingStep(next)
      return next
    })
  }

  const handleOpenRoute = () => {
    navigate(step.route)
  }

  if (!isOpen) {
    if (status === 'completed') {
      return null
    }

    return (
      <button type="button" className={styles.replayButton} onClick={handleOpenGuide}>
        {status === 'dismissed' ? 'Resume guide' : 'Guide'}
      </button>
    )
  }

  return (
    <div className={styles.overlay} role="presentation" onMouseDown={handlePause}>
      <div
        ref={dialogRef}
        className={styles.card}
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
        aria-describedby={descriptionId}
        tabIndex={-1}
        onMouseDown={(event) => event.stopPropagation()}
      >
        <div className={styles.header}>
          <div>
            <div className={styles.eyebrow}>Product guide</div>
            <h2 id={titleId} className={styles.title}>
              {step.title}
            </h2>
          </div>
          <button
            type="button"
            className={styles.closeButton}
            aria-label="Pause product guide"
            onClick={handlePause}
            data-dialog-initial-focus
          >
            ×
          </button>
        </div>

        <div className={styles.progressBlock}>
          <div
            className={styles.progressRow}
            role="progressbar"
            aria-label="Guide progress"
            aria-valuemin={1}
            aria-valuemax={GUIDE_STEPS.length}
            aria-valuenow={stepIndex + 1}
            aria-valuetext={`Step ${stepIndex + 1} of ${GUIDE_STEPS.length}`}
          >
            {GUIDE_STEPS.map((guideStep, index) => (
              <span
                key={guideStep.id}
                className={`${styles.progressDot} ${
                  index === stepIndex ? styles.progressDotActive : ''
                } ${index < stepIndex ? styles.progressDotDone : ''}`}
                aria-hidden="true"
              />
            ))}
          </div>
          <p className={styles.progressCopy} aria-live="polite" aria-atomic="true">
            Step {stepIndex + 1} of {GUIDE_STEPS.length}
          </p>
        </div>

        <div
          key={step.id}
          className={styles.body}
          role="region"
          aria-label={`${step.pageLabel} guide details`}
          tabIndex={0}
          data-testid="onboarding-guide-body"
        >
          <p id={descriptionId} className={styles.description}>
            {step.description}
          </p>

          <div className={styles.detailCard}>
            <div className={styles.detailLabel}>What to do in {step.pageLabel}</div>
            <ul className={styles.detailList}>
              {step.highlights.map((highlight) => (
                <li key={highlight} className={styles.detailItem}>
                  {highlight}
                </li>
              ))}
            </ul>
            {isOnTargetRoute && (
              <div className={styles.detailHint}>You are already on this page.</div>
            )}
          </div>
        </div>

        <div className={styles.footer} data-testid="onboarding-guide-actions">
          <div className={styles.footerLeft}>
            <button type="button" className={styles.secondaryButton} onClick={handlePause}>
              Pause tour
            </button>
          </div>
          <div className={styles.footerRight}>
            {stepIndex > 0 && (
              <button type="button" className={styles.secondaryButton} onClick={handleBack}>
                Back
              </button>
            )}
            {!isOnTargetRoute && (
              <button
                type="button"
                className={styles.secondaryButton}
                onFocus={() => void preloadDashboardRoute(step.route)}
                onPointerDown={() => void preloadDashboardRoute(step.route)}
                onClick={handleOpenRoute}
              >
                {step.actionLabel}
              </button>
            )}
            <button type="button" className={styles.primaryButton} onClick={handleNext}>
              {stepIndex === GUIDE_STEPS.length - 1 ? 'Finish' : 'Next'}
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}

export default OnboardingGuide
