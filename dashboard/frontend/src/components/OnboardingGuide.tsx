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
    description: 'Bring the models you already use into one workspace.',
    highlights: [
      'Choose a local or hosted provider',
      'Connect once with a URL and API key',
      'Import one or many models in a single step',
    ],
    route: '/config/models',
    actionLabel: 'Open Models',
  },
  {
    id: 'mixture',
    pageLabel: 'Mixture-of-Models',
    title: 'Build your model path',
    description: 'Choose a recipe, then assign the right model to each decision.',
    highlights: [
      'Start from a proven recipe',
      'Assign models to its decision paths',
      'Publish one stable model name for applications',
    ],
    route: '/config/entrypoints-recipes',
    actionLabel: 'Build a Mixture',
  },
  {
    id: 'playground',
    pageLabel: 'Playground',
    title: 'Try it in Playground',
    description: 'Send a real prompt and see the selected path as it happens.',
    highlights: [
      'Choose your new Mixture-of-Models',
      'Stream responses through the live router',
      'Reveal the decision, algorithm, and model when needed',
    ],
    route: '/playground',
    actionLabel: 'Open Playground',
  },
  {
    id: 'insights',
    pageLabel: 'Insights',
    title: 'See what you saved',
    description: 'Understand the quality, speed, and cost of every routed request.',
    highlights: [
      'Compare actual spend with your baseline',
      'Inspect the model path behind each result',
      'Use evidence to tune the next version',
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
    // Evaluation is a focused evidence workspace and is not part of this product guide.
    // Keep the launcher from obscuring dense controls or sticky decision actions there.
    if (status === 'completed' || location.pathname === '/evaluation') {
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
          <div className={styles.headerIdentity}>
            <div className={styles.logo} aria-hidden="true">
              <img src="/vllm.png" alt="" />
            </div>
            <div>
              <div className={styles.eyebrow}>Getting started</div>
              <h2 id={titleId} className={styles.title}>
                {step.title}
              </h2>
            </div>
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
