import React, { useState, useEffect } from 'react'
import clsx from 'clsx'
import Link from '@docusaurus/Link'
import Translate, { translate } from '@docusaurus/Translate'
import useDocusaurusContext from '@docusaurus/useDocusaurusContext'
import Layout from '@theme/Layout'
import HomepageFeatures from '@site/src/components/HomepageFeatures'
import ChainOfThoughtTerminal from '@site/src/components/ChainOfThoughtTerminal'

import AcknowledgementsSection from '@site/src/components/AcknowledgementsSection'
import TeamCarousel from '@site/src/components/TeamCarousel'
import TransformerPipelineAnimation from '@site/src/components/TransformerPipelineAnimation'

import styles from './index.module.css'

const ROTATING_WORDS = [
  translate({ id: 'homepage.hero.rotating.modality', message: 'Modality' }),
  translate({ id: 'homepage.hero.rotating.models', message: 'Models' }),
  translate({ id: 'homepage.hero.rotating.tools', message: 'Tools' }),
  translate({ id: 'homepage.hero.rotating.skills', message: 'Skills' }),
]

const HomepageHeader: React.FC = () => {
  const [wordIndex, setWordIndex] = useState(0)
  const [isAnimating, setIsAnimating] = useState(false)

  useEffect(() => {
    const interval = setInterval(() => {
      setIsAnimating(true)
      setTimeout(() => {
        setWordIndex(prev => (prev + 1) % ROTATING_WORDS.length)
        setIsAnimating(false)
      }, 300)
    }, 2500)
    return () => clearInterval(interval)
  }, [])

  return (
    <header className={clsx('hero hero--primary', styles.heroBanner)}>
      <div className="container">
        <div className={styles.heroContent}>
          <div className={styles.heroLeft}>
            {/* Logo Badge */}
            <div className={styles.logoBadge}>
              <img
                src="/img/vllm.png"
                alt="vLLM Logo"
                className={styles.vllmLogoSmall}
              />
              <span className={styles.badgeText}>System Level Intelligence</span>
            </div>

            {/* Main Headline */}
            <h1 className={styles.mainHeadline}>
              <span className={styles.headlineTop}>
                <Translate id="homepage.hero.intelligentRouting">Intelligent Routing</Translate>
              </span>
              <span className={styles.headlineMain}>
                <Translate id="homepage.hero.mixtureOf">for Mixture-of-</Translate>
                <span
                  className={`${styles.rotatingWord} ${isAnimating ? styles.rotatingWordOut : styles.rotatingWordIn}`}
                >
                  {ROTATING_WORDS[wordIndex]}
                </span>
              </span>
            </h1>

            {/* Subtitle */}
            <p className={styles.heroSubtitle}>
              <strong><Translate id="homepage.hero.subtitle.signalDriven">Signal-driven</Translate></strong>
              {' '}
              <Translate id="homepage.hero.subtitle.decisions">decisions</Translate>
              {' · '}
              <strong><Translate id="homepage.hero.subtitle.pluginChain">Plugin-chain</Translate></strong>
              {' '}
              <Translate id="homepage.hero.subtitle.architecture">architecture</Translate>
              <br />
              <Translate id="homepage.hero.subtitle.line2">Cloud · Data Center · Edge</Translate>
            </p>

            {/* Feature Pills */}
            <div className={styles.featurePills}>
              <div className={styles.featurePill}>
                <span className={styles.pillIcon}>🎯</span>
                <span className={styles.pillText}>
                  <Translate id="homepage.hero.pill.signals">Signal-Driven</Translate>
                </span>
              </div>
              <div className={styles.featurePill}>
                <span className={styles.pillIcon}>🔌</span>
                <span className={styles.pillText}>
                  <Translate id="homepage.hero.pill.plugins">Plugin-Chain</Translate>
                </span>
              </div>
              <div className={styles.featurePill}>
                <span className={styles.pillIcon}>🌐</span>
                <span className={styles.pillText}>
                  <Translate id="homepage.hero.pill.deployment">Cloud · DC · Edge</Translate>
                </span>
              </div>
            </div>

            {/* CTA Buttons */}
            <div className={styles.ctaButtons}>
              <Link
                className={styles.primaryButton}
                to="/docs/installation/"
              >
                <span className={styles.buttonText}>
                  <Translate id="homepage.hero.getStarted">Get Started</Translate>
                </span>
                <span className={styles.buttonIcon}>→</span>
              </Link>
              <a
                className={styles.secondaryButton}
                href="https://play.vllm-semantic-router.com/"
                target="_blank"
                rel="noopener noreferrer"
              >
                <span className={styles.buttonText}>
                  <Translate id="homepage.hero.publicBeta">Try Demo</Translate>
                </span>
                <span className={styles.buttonIcon}>↗</span>
              </a>
            </div>
          </div>

          <div className={styles.heroRight}>
            <ChainOfThoughtTerminal />
          </div>
        </div>
      </div>
    </header>
  )
}

const AITechShowcase: React.FC = () => {
  return (
    <section className={styles.aiTechSection}>
      <div className="container">
        <div className={styles.aiTechHeader}>
          <p className={styles.aiTechLabel}>
            <Translate id="homepage.aiTech.label">Built on Encoder Models</Translate>
          </p>
          <h2 className={styles.aiTechTitle}>
            <Translate id="homepage.aiTech.title">Encoder-Based Intelligence</Translate>
          </h2>
          <p className={styles.aiTechDescription}>
            <Translate id="homepage.aiTech.description">
              Purpose-built encoder models extract meaning from every request — understanding intent,
              ranking relevance, and classifying content across modalities in real time.
            </Translate>
          </p>
        </div>

        <TransformerPipelineAnimation />

        <div className={styles.capabilitiesGrid}>
          {/* Row 1: Multi-Modality — wide hero card */}
          <div className={`${styles.capabilityCard} ${styles.capabilityWide}`}>
            <div className={styles.capabilityIcon}>
              <span>🎭</span>
            </div>
            <h3 className={styles.capabilityTitle}>
              <Translate id="homepage.aiTech.cap.multiModality">Multi-Modality</Translate>
            </h3>
            <p className={styles.capabilityDesc}>
              <Translate id="homepage.aiTech.cap.multiModality.desc">
                Detect and route text, image and audio inputs
                to the right modality-capable model.
              </Translate>
            </p>
          </div>

          {/* Row 2: Three equal cards */}
          <div className={styles.capabilityCard}>
            <div className={styles.capabilityIcon}>
              <span>🧬</span>
            </div>
            <h3 className={styles.capabilityTitle}>
              <Translate id="homepage.aiTech.cap.biEncoder">Bi-Encoder Embeddings</Translate>
            </h3>
            <p className={styles.capabilityDesc}>
              <Translate id="homepage.aiTech.cap.biEncoder.desc">
                Independently encode queries and candidates into dense vectors for similarity search and semantic caching.
              </Translate>
            </p>
          </div>

          <div className={styles.capabilityCard}>
            <div className={styles.capabilityIcon}>
              <span>⚡</span>
            </div>
            <h3 className={styles.capabilityTitle}>
              <Translate id="homepage.aiTech.cap.crossEncoder">Cross-Encoder Learning</Translate>
            </h3>
            <p className={styles.capabilityDesc}>
              <Translate id="homepage.aiTech.cap.crossEncoder.desc">
                Joint cross-attention scoring of query-candidate pairs for high-precision reranking.
              </Translate>
            </p>
          </div>

          <div className={styles.capabilityCard}>
            <div className={styles.capabilityIcon}>
              <span>🤔</span>
            </div>
            <h3 className={styles.capabilityTitle}>
              <Translate id="homepage.aiTech.cap.classification">Classification</Translate>
            </h3>
            <p className={styles.capabilityDesc}>
              <Translate id="homepage.aiTech.cap.classification.desc">
                Domain, jailbreak, PII and fact-check classification across 14 MMLU categories via ModernBERT with LoRA.
              </Translate>
            </p>
          </div>

          {/* Row 3: Full Attention + 2DMSE + MRL */}
          <div className={styles.capabilityCard}>
            <div className={styles.capabilityIcon}>
              <span>👁️</span>
            </div>
            <h3 className={styles.capabilityTitle}>
              <Translate id="homepage.aiTech.cap.attention">Full Attention</Translate>
            </h3>
            <p className={styles.capabilityDesc}>
              <Translate id="homepage.aiTech.cap.attention.desc">
                Bidirectional attention across tokens and sentences — full context in both directions, not causal masking.
              </Translate>
            </p>
          </div>

          <div className={styles.capabilityCard}>
            <div className={styles.capabilityIcon}>
              <span>🪆</span>
            </div>
            <h3 className={styles.capabilityTitle}>
              <Translate id="homepage.aiTech.cap.2dmse">2DMSE</Translate>
            </h3>
            <p className={styles.capabilityDesc}>
              <Translate id="homepage.aiTech.cap.2dmse.desc">
                Adjust embedding layers and dimensions at inference time to trade compute for accuracy on the fly.
              </Translate>
            </p>
          </div>

          <div className={styles.capabilityCard}>
            <div className={styles.capabilityIcon}>
              <span>📐</span>
            </div>
            <h3 className={styles.capabilityTitle}>
              <Translate id="homepage.aiTech.cap.mrl">MRL</Translate>
            </h3>
            <p className={styles.capabilityDesc}>
              <Translate id="homepage.aiTech.cap.mrl.desc">
                Truncate embedding vectors to any dimension without retraining — balance accuracy and speed per request.
              </Translate>
            </p>
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
            <Translate id="homepage.architecture.title">Architecture</Translate>
          </h2>
          <div className={styles.architectureImageWrapper}>
            <img
              src="/img/architecture.png"
              alt="Architecture"
              className={styles.architectureImage}
            />
          </div>
        </div>
      </div>
    </section>
  )
}

const WhereItLives: React.FC = () => {
  return (
    <section className={styles.whereItLivesSection}>
      <div className="container">
        <h2 className={styles.whereItLivesTitle}>
          📍
          {' '}
          <Translate id="homepage.whereItLives.title">Where it lives</Translate>
        </h2>
        <p className={styles.whereItLivesSubtitle}>
          <Translate id="homepage.whereItLives.subtitle">It lives between the real world and models</Translate>
        </p>
        <div className={styles.whereItLivesImageWrapper}>
          <img
            src="/img/level.png"
            alt="Where vLLM Semantic Router Lives"
            className={styles.whereItLivesImage}
          />
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
        <HomepageFeatures />
        <div className={styles.connectionSection}>
          <div className={styles.connectionLines}>
            <div className={`${styles.connectionLine} ${styles.connectionLine1}`}></div>
            <div className={`${styles.connectionLine} ${styles.connectionLine2}`}></div>
          </div>
        </div>
        <WhereItLives />
        <div className={styles.connectionSection}>
          <div className={styles.connectionLines}>
            <div className={`${styles.connectionLine} ${styles.connectionLine1}`}></div>
            <div className={`${styles.connectionLine} ${styles.connectionLine2}`}></div>
          </div>
        </div>
        <TeamCarousel />
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
