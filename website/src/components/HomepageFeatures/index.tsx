import React from 'react'
import clsx from 'clsx'
import Translate from '@docusaurus/Translate'
import styles from './styles.module.css'

const HomepageFeatures: React.FC = () => {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className={styles.featuresHeader}>
          <h2 className={styles.featuresTitle}>
            🚀
            {' '}
            <Translate id="features.sectionTitle">Advanced AI Capabilities</Translate>
          </h2>
          <p className={styles.featuresSubtitle}>
            <Translate id="features.sectionSubtitle">Powered by cutting-edge neural networks and machine learning technologies</Translate>
          </p>
        </div>
        <div className="row">
          {/* Intelligent Routing */}
          <div className={clsx('col col--4')}>
            <div className={clsx('card', styles.featureCard)}>
              <div className="text--center padding-horiz--md">
                <h3 className={styles.featureTitle}>
                  <Translate id="features.intelligentRouting.title">🧠 Intelligent Routing</Translate>
                </h3>
                <p className={styles.featureDescription}>
                  <Translate id="features.intelligentRouting.description">Powered by ModernBERT Fine-Tuned Models for intelligent intent understanding, it understands context, intent, and complexity to route requests to the best LLM.</Translate>
                </p>
              </div>
            </div>
          </div>
          {/* AI-Powered Security */}
          <div className={clsx('col col--4')}>
            <div className={clsx('card', styles.featureCard)}>
              <div className="text--center padding-horiz--md">
                <h3 className={styles.featureTitle}>
                  <Translate id="features.aiSecurity.title">🛡️ AI-Powered Security</Translate>
                </h3>
                <p className={styles.featureDescription}>
                  <Translate id="features.aiSecurity.description">Advanced PII Detection and Prompt Guard to identify and block jailbreak attempts, ensuring secure and responsible AI interactions across your infrastructure.</Translate>
                </p>
              </div>
            </div>
          </div>
          {/* Semantic Caching */}
          <div className={clsx('col col--4')}>
            <div className={clsx('card', styles.featureCard)}>
              <div className="text--center padding-horiz--md">
                <h3 className={styles.featureTitle}>
                  <Translate id="features.semanticCaching.title">⚡ Semantic Caching</Translate>
                </h3>
                <p className={styles.featureDescription}>
                  <Translate id="features.semanticCaching.description">Intelligent Similarity Cache that stores semantic representations of prompts, dramatically reducing token usage and latency through smart content matching.</Translate>
                </p>
              </div>
            </div>
          </div>
          {/* Auto-Reasoning Engine */}
          <div className={clsx('col col--4')}>
            <div className={clsx('card', styles.featureCard)}>
              <div className="text--center padding-horiz--md">
                <h3 className={styles.featureTitle}>
                  <Translate id="features.autoReasoning.title">🤖 Auto-Reasoning Engine</Translate>
                </h3>
                <p className={styles.featureDescription}>
                  <Translate id="features.autoReasoning.description">Auto reasoning engine that analyzes request complexity, domain expertise requirements, and performance constraints to automatically select the best model for each task.</Translate>
                </p>
              </div>
            </div>
          </div>
          {/* Real-time Analytics */}
          <div className={clsx('col col--4')}>
            <div className={clsx('card', styles.featureCard)}>
              <div className="text--center padding-horiz--md">
                <h3 className={styles.featureTitle}>
                  <Translate id="features.analytics.title">🔬 Real-time Analytics</Translate>
                </h3>
                <p className={styles.featureDescription}>
                  <Translate id="features.analytics.description">Comprehensive monitoring and analytics dashboard with neural network insights, model performance metrics, and intelligent routing decisions visualization.</Translate>
                </p>
              </div>
            </div>
          </div>
          {/* Scalable Architecture */}
          <div className={clsx('col col--4')}>
            <div className={clsx('card', styles.featureCard)}>
              <div className="text--center padding-horiz--md">
                <h3 className={styles.featureTitle}>
                  <Translate id="features.scalable.title">🚀 Scalable Architecture</Translate>
                </h3>
                <p className={styles.featureDescription}>
                  <Translate id="features.scalable.description">Cloud-native design with distributed neural processing, auto-scaling capabilities, and seamless integration with existing LLM infrastructure and model serving platforms.</Translate>
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}

export default HomepageFeatures
