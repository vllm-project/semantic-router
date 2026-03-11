import React from 'react'
import { useNavigate } from 'react-router-dom'
import ColorBends from '../components/ColorBends'
import styles from './LandingPage.module.css'

const LandingPage: React.FC = () => {
  const navigate = useNavigate()
  return (
    <div className={styles.container}>
      <div className={styles.backgroundEffect}>
        <ColorBends
          colors={['#76b900', '#00b4d8', '#ffffff']}
          rotation={20}
          speed={0.2}
          scale={1}
          frequency={1}
          warpStrength={1}
          mouseInfluence={1}
          parallax={0.5}
          noise={0.08}
          transparent
          autoRotate={0.8}
        />
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
              onClick={() => navigate('https://vllm-semantic-router.com')}
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
