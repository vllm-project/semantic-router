import React, { Suspense, lazy } from 'react'
import { useNavigate } from 'react-router-dom'
import styles from './LandingPage.module.css'

const ColorBends = lazy(() => import('../components/ColorBends'))

const LandingPage: React.FC = () => {
  const navigate = useNavigate()
  return (
    <div className={styles.container}>
      <div className={styles.backgroundEffect}>
        <Suspense fallback={null}>
          <ColorBends
            colors={['#f5f5f7', '#e31b23', '#5f636a']}
            rotation={138}
            speed={0.08}
            scale={1}
            frequency={1}
            warpStrength={0.55}
            mouseInfluence={0.35}
            parallax={0.22}
            noise={0.04}
            transparent
            autoRotate={0.22}
          />
        </Suspense>
      </div>

      {/* Main Content - Centered */}
      <main className={styles.mainContent}>
        <div className={styles.heroSection}>
          <div className={styles.heroBadge}>
            <img src="/vllm.png" alt="vLLM Logo" className={styles.badgeLogo} />
            <span>Powered by vLLM Semantic Router</span>
          </div>

          <h1 className={styles.title}>
            <span>Extract signals</span>
            <span className={styles.titleAccent}>Compose decisions.</span>
            <span>Route the best model.</span>
          </h1>

          <p className={styles.subtitle}>
            The System Level Intelligence for <span className={styles.highlight}>Mixture-of-Modality</span>{' '}
            Models.
          </p>

          <p className={styles.deployTargets}>
            Cloud · Data Center · Edge
          </p>

          <div className={styles.ctaGroup}>
            <button
              className={styles.primaryButton}
              onClick={() => navigate('/login')}
            >
              Get Started
            </button>
            <button
              className={styles.secondaryButton}
              onClick={() => window.open('https://vllm-semantic-router.com', '_blank', 'noopener,noreferrer')}
            >
              Learn More
            </button>
          </div>
        </div>
      </main>
    </div>
  )
}

export default LandingPage
