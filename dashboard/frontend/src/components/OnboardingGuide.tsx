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
import ProductIcon from './ProductIcon'

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
    title: 'Connect your models',
    description: 'Add the inference endpoints that will power your model paths.',
    highlights: [
      'Choose a provider and connect its API',
      'Import one or many available models',
      'Verify every endpoint before routing traffic',
    ],
    route: '/config/models',
    actionLabel: 'Open Models',
  },
  {
    id: 'mixture',
    pageLabel: 'Mixture-of-Models',
    title: 'Build a Mixture-of-Models',
    description: 'Start from a recipe, then assign the right models to each decision.',
    highlights: [
      'Choose or create a reusable recipe',
      'Connect signals, projections, decisions, and algorithms',
      'Publish a stable model name for applications',
    ],
    route: '/config/entrypoints-recipes',
    actionLabel: 'Open Mixture-of-Models',
  },
  {
    id: 'playground',
    pageLabel: 'Playground',
    title: 'Test your model path',
    description: 'Run a real prompt and watch the router choose the path.',
    highlights: [
      'Choose a Mixture-of-Models',
      'Follow the selected decision, algorithm, and model',
      'Try tools and real conversation turns',
    ],
    route: '/playground',
    actionLabel: 'Open Playground',
  },
  {
    id: 'access',
    pageLabel: 'Access',
    title: 'Give your team access',
    description: 'Control who can use each model path and how much capacity they receive.',
    highlights: [
      'Create a team and invite its members',
      'Issue API keys with model grants',
      'Apply RPM, TPM, and daily token budgets',
    ],
    route: '/access/teams',
    actionLabel: 'Open Access',
  },
  {
    id: 'insights',
    pageLabel: 'Insights',
    title: 'See what the router saved',
    description: 'Turn every routed request into a clear cost and model-selection story.',
    highlights: [
      'Compare actual spend with the baseline',
      'Inspect model, decision, and signal mix',
      'Replay individual requests when something looks wrong',
    ],
    route: '/insights',
    actionLabel: 'Open Insights',
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
      <button
        type="button"
        className={styles.replayButton}
        onClick={handleOpenGuide}
        aria-label={status === 'dismissed' ? 'Resume product guide' : 'Open product guide'}
        title={status === 'dismissed' ? 'Resume guide' : 'Product guide'}
      >
        <span aria-hidden="true">?</span>
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
            <ProductIcon name="close" />
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
