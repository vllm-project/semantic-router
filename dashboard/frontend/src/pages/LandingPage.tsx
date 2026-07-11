import React, { Suspense, lazy } from 'react'
import { useNavigate } from 'react-router-dom'
import {
  DASHBOARD_COLOR_BENDS_MOTION,
  DASHBOARD_MOTION_COLORS,
} from '../components/dashboardMotionTheme'
import PublicFooter from '../components/PublicFooter'
import PublicHeader from '../components/PublicHeader'
import styles from './LandingPage.module.css'

const ColorBends = lazy(() => import('../components/ColorBends'))

const LandingPage: React.FC = () => {
  const navigate = useNavigate()
  return (
    <div className={styles.container}>
      <div className={styles.backgroundEffect} data-testid="landing-motion-background">
        <Suspense fallback={null}>
          <ColorBends
            colors={DASHBOARD_MOTION_COLORS}
            {...DASHBOARD_COLOR_BENDS_MOTION}
            transparent
          />
        </Suspense>
      </div>

      <PublicHeader />

      <main className={styles.mainContent}>
        <section className={styles.heroSection}>
          <div className={styles.heroBadge}>Open-source runtime for Mixture-of-Models</div>

          <h1 className={styles.title}>
            <span className={styles.titleAccent}>Intelligence,</span>
            <span>composed for you.</span>
          </h1>

          <p className={styles.subtitle}>
            One model or many, across compute and locations—shaped by your priorities.
          </p>

          <div className={styles.ctaGroup}>
            <button className={styles.primaryButton} onClick={() => navigate('/login')}>
              Enter Dashboard
            </button>
            <button
              className={styles.secondaryButton}
              onClick={() =>
                window.open(
                  'https://vllm-semantic-router.com/docs/intro/',
                  '_blank',
                  'noopener,noreferrer',
                )
              }
            >
              Explore the Docs
            </button>
          </div>
        </section>

        <section className={styles.routingSection} aria-labelledby="routing-section-title">
          <div className={styles.sectionHeading}>
            <span>Signal-driven routing</span>
            <h2 id="routing-section-title">From request to model path.</h2>
          </div>

          <div className={styles.routingGrid}>
            <article className={styles.routingStep}>
              <span className={styles.stepIndex}>01</span>
              <h3>Extract signals</h3>
              <p>Read intent, safety, context, and modality before generation begins.</p>
            </article>
            <article className={styles.routingStep}>
              <span className={styles.stepIndex}>02</span>
              <h3>Compose decisions</h3>
              <p>Turn heterogeneous signals into policies shaped by your priorities.</p>
            </article>
            <article className={styles.routingStep}>
              <span className={styles.stepIndex}>03</span>
              <h3>Route one—or coordinate many</h3>
              <p>Select model paths across compute and locations without changing your API.</p>
            </article>
          </div>
        </section>
      </main>

      <PublicFooter />
    </div>
  )
}

export default LandingPage
