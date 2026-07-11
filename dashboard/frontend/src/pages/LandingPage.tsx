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
          <div className={styles.heroBadge}>The next-generation model architecture</div>

          <h1 className={styles.title}>
            <span className={styles.titleAccent}>Build your</span>
            <span>Mixture-of-Models.</span>
          </h1>

          <p className={styles.subtitle}>
            Compose a model system around every user, product, and workload. vLLM Semantic Router
            turns preferences and request signals into executable model paths.
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
            <span>Signal-driven composition</span>
            <h2 id="routing-section-title">From signals and preference to model path.</h2>
          </div>

          <div className={styles.routingGrid}>
            <article className={styles.routingStep}>
              <span className={styles.stepIndex}>01</span>
              <h3>Understand every request</h3>
              <p>Extract intent, context, safety, modality, and feedback before generation.</p>
            </article>
            <article className={styles.routingStep}>
              <span className={styles.stepIndex}>02</span>
              <h3>Make preference executable</h3>
              <p>Turn user, product, and workload preferences into executable policy.</p>
            </article>
            <article className={styles.routingStep}>
              <span className={styles.stepIndex}>03</span>
              <h3>Compose the model path</h3>
              <p>Route, cascade, or fuse across the models that fit each request.</p>
            </article>
          </div>
        </section>
      </main>

      <PublicFooter />
    </div>
  )
}

export default LandingPage
