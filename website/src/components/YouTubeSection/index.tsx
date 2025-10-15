import React from 'react'
import styles from './styles.module.css'

const YouTubeSection: React.FC = () => {
  return (
    <section className={styles.youtubeSection}>
      <div className="container">
        <div className={styles.youtubeContainer}>
          <h2 className={styles.youtubeTitle}>
            🎥 vLLM Semantic Router Demos
          </h2>
          <p className={styles.youtubeDescription}>
            <strong>Latest News</strong>
            {' '}
            🎉: User Experience is something we do care about. Introducing vLLM-SR dashboard:
          </p>
          <div className={styles.featureList}>
            <div className={styles.featureItem}>
              <span className={styles.featureIcon}>💬</span>
              <span>Chat with vLLM-SR and see its thinking chain</span>
            </div>
            <div className={styles.featureItem}>
              <span className={styles.featureIcon}>🗺️</span>
              <span>View the topology of the intents for Models</span>
            </div>
            <div className={styles.featureItem}>
              <span className={styles.featureIcon}>📊</span>
              <span>Monitor real-time Metrics with Grafana Dashboard</span>
            </div>
            <div className={styles.featureItem}>
              <span className={styles.featureIcon}>⚙️</span>
              <span>Configure Mixture-of-Models with different Domains</span>
            </div>
          </div>
          <div className={styles.videoWrapper}>
            <iframe
              className={styles.videoIframe}
              src="https://www.youtube.com/embed/E2IirN8PsFw"
              title="vLLM Semantic Router Dashboard"
              allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
              allowFullScreen
            />
          </div>
        </div>
      </div>
    </section>
  )
}

export default YouTubeSection
