import React, { useEffect, useState } from 'react'
import { useLocation, useNavigate } from 'react-router-dom'
import {
  getOnboardingStatus,
  markOnboardingPending,
  setOnboardingStatus,
} from '../utils/onboarding'
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
      'This is where signals become decisions, priorities, and plugins that shape how requests move through the router.',
    highlights: [
      'Apply a preset or build rules manually',
      'Use priorities to decide which route wins first',
      'Review plugin effects before promoting changes',
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
      'Check whether the chosen preset behaves as expected',
      'Iterate here before changing real traffic',
    ],
    route: '/playground',
    actionLabel: 'Open Playground',
  },
  {
    id: 'dsl',
    pageLabel: 'DSL Builder',
    title: 'Author router behavior directly in DSL',
    description:
      'Use Builder when presets stop being expressive enough.',
    highlights: [
      'Open the Guide drawer for DSL snippets',
      'Author signals, routes, plugins, and backends',
      'Compile and deploy deeper routing changes',
    ],
    route: '/builder',
    actionLabel: 'Open DSL Builder',
  },
  {
    id: 'clawos',
    pageLabel: 'ClawOS',
    title: 'Orchestrate multi-claw worker systems',
    description:
      'Use ClawOS when one router needs multi-agent orchestration.',
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

  useEffect(() => {
    const status = getOnboardingStatus()
    setIsOpen(status === 'pending')
    setIsReady(true)
  }, [])

  if (!isReady || location.pathname === '/') {
    return null
  }

  const step = GUIDE_STEPS[stepIndex]
  const isOnTargetRoute = location.pathname === step.route

  const handleReplay = () => {
    markOnboardingPending()
    setStepIndex(0)
    setIsOpen(true)
  }

  const handleSkip = () => {
    setOnboardingStatus('dismissed')
    setIsOpen(false)
  }

  const handleNext = () => {
    if (stepIndex === GUIDE_STEPS.length - 1) {
      setOnboardingStatus('completed')
      setIsOpen(false)
      return
    }

    setStepIndex((current) => current + 1)
  }

  const handleBack = () => {
    setStepIndex((current) => (current === 0 ? current : current - 1))
  }

  const handleOpenRoute = () => {
    navigate(step.route)
  }

  if (!isOpen) {
    return (
      <button className={styles.replayButton} onClick={handleReplay}>
        Guide
      </button>
    )
  }

  return (
    <div className={styles.overlay}>
      <div className={styles.card}>
        <div className={styles.header}>
          <div>
            <div className={styles.eyebrow}>Product guide</div>
            <h2 className={styles.title}>{step.title}</h2>
          </div>
          <button className={styles.closeButton} onClick={handleSkip}>
            ×
          </button>
        </div>

        <div className={styles.progressRow}>
          {GUIDE_STEPS.map((guideStep, index) => (
            <span
              key={guideStep.id}
              className={`${styles.progressDot} ${
                index === stepIndex ? styles.progressDotActive : ''
              } ${index < stepIndex ? styles.progressDotDone : ''}`}
            />
          ))}
        </div>

        <p className={styles.description}>{step.description}</p>

        <div className={styles.detailCard}>
          <div className={styles.detailLabel}>What to do in {step.pageLabel}</div>
          <ul className={styles.detailList}>
            {step.highlights.map((highlight) => (
              <li key={highlight} className={styles.detailItem}>
                {highlight}
              </li>
            ))}
          </ul>
          {isOnTargetRoute && <div className={styles.detailHint}>You are already on this page.</div>}
        </div>

        <div className={styles.footer}>
          <div className={styles.footerLeft}>
            <button className={styles.secondaryButton} onClick={handleSkip}>
              Skip tour
            </button>
          </div>
          <div className={styles.footerRight}>
            {stepIndex > 0 && (
              <button className={styles.secondaryButton} onClick={handleBack}>
                Back
              </button>
            )}
            {!isOnTargetRoute && (
              <button className={styles.secondaryButton} onClick={handleOpenRoute}>
                {step.actionLabel}
              </button>
            )}
            <button className={styles.primaryButton} onClick={handleNext}>
              {stepIndex === GUIDE_STEPS.length - 1 ? 'Finish' : 'Next'}
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}

export default OnboardingGuide
