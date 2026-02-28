import React from 'react'
import { useNavigate } from 'react-router-dom'
import PrismaticBurst from '../components/PrismaticBurst'
import styles from './LandingPage.module.css'

const LandingPage: React.FC = () => {
  const navigate = useNavigate()

  return (
    <div className={styles.container}>
      <div className={styles.backgroundEffect}>
        <PrismaticBurst
          animationType="rotate3d"
          intensity={2}
          speed={0.5}
          distort={0}
          paused={false}
          offset={{ x: 0, y: 0 }}
          hoverDampness={0.25}
          rayCount={0}
          mixBlendMode="lighten"
          colors={['#76b900', '#00b4d8', '#ffffff']}
        />
      </div>

      {/* Main Content - Centered */}
      <main className={styles.mainContent}>
        <div className={styles.heroSection}>
          <h1 className={styles.title}>
            <img src="/vllm.png" alt="vLLM Logo" className={styles.logoInline} />
            LLM Semantic Router
          </h1>

          <p className={styles.subtitle}>
            System Level Intelligence for{' '}
            <span className={styles.highlight}>Mixture-of-Models</span>
          </p>
          <p className={styles.deployTargets}>
            Cloud · Data Center · Edge
          </p>

          <button
            className={styles.launchButton}
            onClick={() => navigate('/dashboard')}
          >
            <span className={styles.launchText}>Launch</span>
            <div className={styles.launchGlow}></div>
          </button>
        </div>
      </main>
    </div>
  )
}

export default LandingPage
