import React from 'react'
import clsx from 'clsx'
import Link from '@docusaurus/Link'
import Translate from '@docusaurus/Translate'
import useDocusaurusContext from '@docusaurus/useDocusaurusContext'
import Layout from '@theme/Layout'
import HomepageFeatures from '@site/src/components/HomepageFeatures'
import ChainOfThoughtTerminal from '@site/src/components/ChainOfThoughtTerminal'
import NeuralNetworkBackground from '@site/src/components/NeuralNetworkBackground'
import AIChipAnimation from '@site/src/components/AIChipAnimation'
import AcknowledgementsSection from '@site/src/components/AcknowledgementsSection'
import YouTubeSection from '@site/src/components/YouTubeSection'

import styles from './index.module.css'

const HomepageHeader: React.FC = () => {
  return (
    <header className={clsx('hero hero--primary', styles.heroBanner)}>
      <NeuralNetworkBackground />
      <div className="container">
        <div className={styles.heroContent}>
          <div className={styles.heroLeft}>
            <div className={styles.heroTitle}>
              <img
                src="/img/vllm.png"
                alt="vLLM Logo"
                className={styles.vllmLogo}
              />
              <h1 className="hero__title">
                <span className={styles.aiGlow}>
                  <Translate id="homepage.hero.aiPowered">AI-Powered</Translate>
                </span>
                {' '}
                vLLM Semantic Router
              </h1>
            </div>
            <p className="hero__subtitle">
              🧠
              {' '}
              <Translate id="homepage.hero.intelligent">Intelligent</Translate>
              {' '}
              <strong><Translate id="homepage.hero.autoReasoning">Auto Reasoning</Translate></strong>
              {' '}
              <Translate id="homepage.hero.routerFor">Router for Efficient LLM Inference on</Translate>
              {' '}
              <strong><Translate id="homepage.hero.mixtureOfModels">Mixture-of-Models</Translate></strong>
              <br />
              <span className={styles.techBadges}>
                <span className={styles.techBadge}>
                  🧬
                  <Translate id="homepage.hero.badge.neural">Neural Networks</Translate>
                </span>
                <span className={styles.techBadge}>
                  ⚡
                  <Translate id="homepage.hero.badge.llm">LLM Optimization</Translate>
                </span>
                <span className={styles.techBadge}>
                  ♻️
                  <Translate id="homepage.hero.badge.economics">Per-token Unit Economics</Translate>
                </span>
              </span>
            </p>
          </div>
          <div className={styles.heroRight}>
            <ChainOfThoughtTerminal />
          </div>
        </div>
        <div className={styles.buttons}>
          <Link
            className="button button--secondary button--lg"
            to="/docs/intro"
          >
            🚀
            {' '}
            <Translate id="homepage.hero.getStarted">Get Started - 5min ⏱️</Translate>
          </Link>
        </div>
      </div>
    </header>
  )
}

const AITechShowcase: React.FC = () => {
  return (
    <section className={styles.aiTechSection}>
      <div className="container">
        <div className={styles.aiTechContainer}>
          <div className={styles.aiTechLeft}>
            <h2 className={styles.aiTechTitle}>
              🧠
              {' '}
              <Translate id="homepage.aiTech.title">Neural Processing Architecture</Translate>
            </h2>
            <p className={styles.aiTechDescription}>
              <Translate id="homepage.aiTech.description">
                Powered by cutting-edge AI technologies including ModernBERT fine-tuned models,
                and advanced semantic understanding for intelligent
                model routing and selection.
              </Translate>
            </p>
            <div className={styles.aiFeatures}>
              <div className={styles.aiFeature}>
                <span className={styles.aiFeatureIcon}>🤖</span>
                <span><Translate id="homepage.aiTech.feature.slm">Small Language Models</Translate></span>
              </div>
              <div className={styles.aiFeature}>
                <span className={styles.aiFeatureIcon}>🧬</span>
                <span><Translate id="homepage.aiTech.feature.neural">Neural Network Processing</Translate></span>
              </div>
              <div className={styles.aiFeature}>
                <span className={styles.aiFeatureIcon}>⚡</span>
                <span><Translate id="homepage.aiTech.feature.inference">Real-time Inference</Translate></span>
              </div>
              <div className={styles.aiFeature}>
                <span className={styles.aiFeatureIcon}>🎯</span>
                <span><Translate id="homepage.aiTech.feature.semantic">Semantic Understanding</Translate></span>
              </div>
            </div>
          </div>
          <div className={styles.aiTechRight}>
            <AIChipAnimation />
          </div>
        </div>
      </div>
    </section>
  )
}

const FlowDiagram: React.FC = () => {
  return (
    <section className={styles.flowSection}>
      <div className="container">
        <div className={styles.architectureContainer}>
          <h2 className={styles.architectureTitle}>
            🏗️
            {' '}
            <Translate id="homepage.architecture.title">Intent Aware Semantic Router Architecture</Translate>
          </h2>
          <div className={styles.architectureImageWrapper}>
            <img
              src="/img/architecture.png"
              alt="Intent Aware Semantic Router Architecture"
              className={styles.architectureImage}
            />
          </div>
        </div>
      </div>
    </section>
  )
}

const Home: React.FC = () => {
  const { siteConfig } = useDocusaurusContext()
  return (
    <Layout
      title={`${siteConfig.title}`}
      description="AI-Powered Intelligent Mixture-of-Models Router with Neural Network Processing"
    >
      <HomepageHeader />
      <main>
        <AITechShowcase />
        <div className={styles.connectionSection}>
          <div className={styles.connectionLines}>
            <div className={`${styles.connectionLine} ${styles.connectionLine1}`}></div>
            <div className={`${styles.connectionLine} ${styles.connectionLine2}`}></div>
            <div className={`${styles.connectionLine} ${styles.connectionLine3}`}></div>
          </div>
        </div>
        <FlowDiagram />
        <div className={styles.connectionSection}>
          <div className={styles.connectionLines}>
            <div className={`${styles.connectionLine} ${styles.connectionLine4}`}></div>
            <div className={`${styles.connectionLine} ${styles.connectionLine5}`}></div>
          </div>
        </div>
        <YouTubeSection />
        <div className={styles.connectionSection}>
          <div className={styles.connectionLines}>
            <div className={`${styles.connectionLine} ${styles.connectionLine1}`}></div>
            <div className={`${styles.connectionLine} ${styles.connectionLine2}`}></div>
          </div>
        </div>
        <HomepageFeatures />
        <div className={styles.connectionSection}>
          <div className={styles.connectionLines}>
            <div className={`${styles.connectionLine} ${styles.connectionLine1}`}></div>
            <div className={`${styles.connectionLine} ${styles.connectionLine2}`}></div>
          </div>
        </div>
        <AcknowledgementsSection />
      </main>
    </Layout>
  )
}

export default Home
