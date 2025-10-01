import React from 'react'
import clsx from 'clsx'
import styles from './styles.module.css'

interface Feature {
  title: string
  description: React.ReactNode
}

const FeatureList: Feature[] = [
  {
    title: '🧠 Intelligent Routing',
    description: (
      <>
        Powered by
        {' '}
        <strong>ModernBERT Fine-Tuned Models</strong>
        {' '}
        for
        intelligent intent understanding, it understands context, intent,
        and complexity to route requests to the best LLM.
      </>
    ),
  },
  {
    title: '🛡️ AI-Powered Security',
    description: (
      <>
        Advanced
        {' '}
        <strong>PII Detection</strong>
        {' '}
        and
        {' '}
        <strong>Prompt Guard</strong>
        {' '}
        to identify and block jailbreak attempts, ensuring secure and responsible AI interactions
        across your infrastructure.
      </>
    ),
  },
  {
    title: '⚡ Semantic Caching',
    description: (
      <>
        Intelligent
        {' '}
        <strong>Similarity Cache</strong>
        {' '}
        that stores semantic representations
        of prompts, dramatically reducing token usage and latency through smart content matching.
      </>
    ),
  },
  {
    title: '🤖 Auto-Reasoning Engine',
    description: (
      <>
        Auto reasoning engine that analyzes request
        {' '}
        <strong>complexity</strong>
        , domain expertise
        requirements, and performance constraints to automatically select the best model for each task.
      </>
    ),
  },
  {
    title: '🔬 Real-time Analytics',
    description: (
      <>
        Comprehensive monitoring and analytics dashboard with
        {' '}
        <strong>neural network insights</strong>
        ,
        model performance metrics, and intelligent routing decisions visualization.
      </>
    ),
  },
  {
    title: '🚀 Scalable Architecture',
    description: (
      <>
        Cloud-native design with
        {' '}
        <strong>distributed neural processing</strong>
        , auto-scaling capabilities,
        and seamless integration with existing LLM infrastructure and model serving platforms.
      </>
    ),
  },
]

const Feature: React.FC<Feature> = ({ title, description }) => {
  return (
    <div className={clsx('col col--4')}>
      <div className={clsx('card', styles.featureCard)}>
        <div className="text--center padding-horiz--md">
          <h3 className={styles.featureTitle}>{title}</h3>
          <p className={styles.featureDescription}>{description}</p>
        </div>
      </div>
    </div>
  )
}

const HomepageFeatures: React.FC = () => {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className={styles.featuresHeader}>
          <h2 className={styles.featuresTitle}>
            🚀 Advanced AI Capabilities
          </h2>
          <p className={styles.featuresSubtitle}>
            Powered by cutting-edge neural networks and machine learning technologies
          </p>
        </div>
        <div className="row">
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  )
}

export default HomepageFeatures
