import React from 'react'
import Translate from '@docusaurus/Translate'
import styles from './styles.module.css'

const YouTubeSection: React.FC = () => {
  return (
    <section className={styles.youtubeSection}>
      <div className="container">
        <div className={styles.youtubeContainer}>
          <h2 className={styles.youtubeTitle}>
            🎥
            {' '}
            <Translate id="youtube.title">vLLM Semantic Router Demos</Translate>
          </h2>
          <p className={styles.youtubeDescription}>
            <strong><Translate id="youtube.latestNews">Latest News</Translate></strong>
            {' '}
            🎉:
            {' '}
            <Translate id="youtube.description">User Experience is something we do care about. Introducing vLLM-SR dashboard:</Translate>
          </p>
          <div className={styles.featureList}>
            <div className={styles.featureItem}>
              <span className={styles.featureIcon}>💬</span>
              <span><Translate id="youtube.feature.chat">Chat with vLLM-SR and see its thinking chain</Translate></span>
            </div>
            <div className={styles.featureItem}>
              <span className={styles.featureIcon}>🗺️</span>
              <span><Translate id="youtube.feature.topology">View the topology of the intents for Models</Translate></span>
            </div>
            <div className={styles.featureItem}>
              <span className={styles.featureIcon}>📊</span>
              <span><Translate id="youtube.feature.metrics">Monitor real-time Metrics with Grafana Dashboard</Translate></span>
            </div>
            <div className={styles.featureItem}>
              <span className={styles.featureIcon}>⚙️</span>
              <span><Translate id="youtube.feature.configure">Configure Mixture-of-Models with different Domains</Translate></span>
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
