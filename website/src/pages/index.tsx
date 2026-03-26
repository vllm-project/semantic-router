import React from 'react'
import Layout from '@theme/Layout'
import Translate, { translate } from '@docusaurus/Translate'
import useDocusaurusContext from '@docusaurus/useDocusaurusContext'
import AcknowledgementsSection from '@site/src/components/AcknowledgementsSection'
import InstallQuickStartSection from '@site/src/components/InstallQuickStartSection'
import PaperFigureShowcase from '@site/src/components/PaperFigureShowcase'
import ResearchPaperCarousel from '@site/src/components/ResearchPaperCarousel'
import TeamCarousel from '@site/src/components/TeamCarousel'
import { researchPapers } from '@site/src/data/researchContent'
import TransformerPipelineAnimation from '@site/src/components/TransformerPipelineAnimation'
import CapabilityGlyph, { type CapabilityGlyphKind } from '@site/src/components/site/CapabilityGlyph'
import DitherField from '@site/src/components/site/DitherField'
import { PageIntro, PillLink, SectionLabel, StatStrip } from '@site/src/components/site/Chrome'
import styles from './index.module.css'

const paperCount = researchPapers.length

const heroStats = [
  {
    label: translate({ id: 'homepage.stats.signals.label', message: 'Signals' }),
    value: '14',
    description: translate({
      id: 'homepage.stats.signals.description',
      message: '14 signal families across 5 heuristic and 9 learned detectors.',
    }),
  },
  {
    label: translate({ id: 'homepage.stats.algorithms.label', message: 'Selection' }),
    value: '12',
    description: translate({
      id: 'homepage.stats.algorithms.description',
      message: '12 selectors across symbolic policy, latency heuristics, reinforcement learning, and ML routing.',
    }),
  },
  {
    label: translate({ id: 'homepage.stats.papers.label', message: 'Papers' }),
    value: String(paperCount).padStart(2, '0'),
    description: translate(
      {
        id: 'homepage.stats.papers.description',
        message: '{count} research papers spanning routing, systems, safety, and multimodality.',
      },
      { count: paperCount },
    ),
  },
]

interface CapabilityCard {
  kind: CapabilityGlyphKind
  text: string
  title: string
}

const capabilityCards: CapabilityCard[] = [
  {
    kind: 'signal',
    title: translate({ id: 'homepage.capabilities.signal.title', message: 'Signal extraction' }),
    text: translate({
      id: 'homepage.capabilities.signal.text',
      message: 'Heuristic and learned detectors turn raw requests into typed routing state.',
    }),
  },
  {
    kind: 'projection',
    title: translate({ id: 'homepage.capabilities.projection.title', message: 'Projection coordination' }),
    text: translate({
      id: 'homepage.capabilities.projection.text',
      message: 'Partitions, scores, and mappings coordinate matched evidence into reusable routing facts.',
    }),
  },
  {
    kind: 'decision',
    title: translate({ id: 'homepage.capabilities.decision.title', message: 'Decision engine' }),
    text: translate({
      id: 'homepage.capabilities.decision.text',
      message: 'Signals and projection outputs meet symbolic rules in auditable routing logic.',
    }),
  },
  {
    kind: 'plugin',
    title: translate({ id: 'homepage.capabilities.plugins.title', message: 'Plugin chain' }),
    text: translate({
      id: 'homepage.capabilities.plugins.text',
      message: 'Cache, safety, rewrite, and tracing attach as composable behaviors.',
    }),
  },
  {
    kind: 'selection',
    title: translate({ id: 'homepage.capabilities.research.title', message: 'Frontier LLM systems' }),
    text: translate({
      id: 'homepage.capabilities.research.text',
      message: 'Research drives the stack itself, exploring frontier LLM systems beyond settled patterns.',
    }),
  },
  {
    kind: 'docs',
    title: translate({ id: 'homepage.capabilities.docs.title', message: 'Full dashboard support' }),
    text: translate({
      id: 'homepage.capabilities.docs.text',
      message: 'Operate routing, topology, controls, and runtime feedback from one integrated dashboard.',
    }),
  },
]

const encoderTracks = [
  {
    label: 'SEQ_CLS',
    text: translate({
      id: 'homepage.aiTech.track.sequence',
      message: 'Sequence classification for domain, jailbreak, fact-check, and feedback routing.',
    }),
  },
  {
    label: 'TOKEN',
    text: translate({
      id: 'homepage.aiTech.track.token',
      message: 'Token labeling for PII and safety-sensitive spans that need localized intervention.',
    }),
  },
  {
    label: 'EMBED',
    text: translate({
      id: 'homepage.aiTech.track.embedding',
      message: 'Embedding and rerank paths for semantic cache, similarity search, and candidate scoring.',
    }),
  },
]

const encoderSpotlightCard = {
  marker: 'MOD',
  title: translate({ id: 'homepage.aiTech.cap.multiModality', message: 'Multi-Modality' }),
  text: translate({
    id: 'homepage.aiTech.cap.multiModality.desc',
    message: 'Detect and route text, image and audio inputs to the right modality-capable model.',
  }),
}

const encoderCards = [
  {
    marker: 'BIE',
    title: translate({ id: 'homepage.aiTech.cap.biEncoder', message: 'Bi-Encoder Embeddings' }),
    text: translate({
      id: 'homepage.aiTech.cap.biEncoder.desc',
      message: 'Independently encode queries and candidates into dense vectors for similarity search and semantic caching.',
    }),
  },
  {
    marker: 'XCE',
    title: translate({ id: 'homepage.aiTech.cap.crossEncoder', message: 'Cross-Encoder Learning' }),
    text: translate({
      id: 'homepage.aiTech.cap.crossEncoder.desc',
      message: 'Joint cross-attention scoring of query-candidate pairs for high-precision reranking.',
    }),
  },
  {
    marker: 'CLS',
    title: translate({ id: 'homepage.aiTech.cap.classification', message: 'Classification' }),
    text: translate({
      id: 'homepage.aiTech.cap.classification.desc',
      message: 'Domain, jailbreak, PII and fact-check classification across 14 MMLU categories via ModernBERT with LoRA.',
    }),
  },
  {
    marker: 'ATT',
    title: translate({ id: 'homepage.aiTech.cap.attention', message: 'Full Attention' }),
    text: translate({
      id: 'homepage.aiTech.cap.attention.desc',
      message: 'Bidirectional attention across tokens and sentences, with full context instead of causal masking.',
    }),
  },
  {
    marker: '2DM',
    title: translate({ id: 'homepage.aiTech.cap.2dmse', message: '2DMSE' }),
    text: translate({
      id: 'homepage.aiTech.cap.2dmse.desc',
      message: 'Adjust embedding layers and dimensions at inference time to trade compute for accuracy on the fly.',
    }),
  },
  {
    marker: 'MRL',
    title: translate({ id: 'homepage.aiTech.cap.mrl', message: 'MRL' }),
    text: translate({
      id: 'homepage.aiTech.cap.mrl.desc',
      message: 'Truncate embedding vectors to any dimension without retraining to balance accuracy and speed per request.',
    }),
  },
]

function DitherHero(): JSX.Element {
  return (
    <header className={styles.hero}>
      <DitherField className={styles.heroNoise} />
      <div className="site-shell-container">
        <div className={styles.heroGrid}>
          <div className={styles.heroStack}>
            <PageIntro
              className={styles.heroIntro}
              label={<Translate id="homepage.hero.label">System-level intelligence</Translate>}
              title={(
                <>
                  <Translate id="homepage.hero.line1">Signal</Translate>
                  <br />
                  <Translate id="homepage.hero.line2">before scale</Translate>
                </>
              )}
              actions={(
                <>
                  <PillLink
                    className={styles.heroPrimaryCta}
                    href="https://play.vllm-semantic-router.com/"
                    rel="noreferrer"
                    target="_blank"
                  >
                    <Translate id="homepage.hero.publicBeta">Public Beta</Translate>
                  </PillLink>
                  <PillLink className={styles.heroSecondaryCta} to="/white-paper" muted>
                    <Translate id="homepage.hero.secondaryCta">Open white paper</Translate>
                  </PillLink>
                </>
              )}
            />

            <p className={styles.heroManifesto}>
              <span className={styles.heroManifestoText}>
                <Translate id="homepage.hero.manifesto">
                  System-brain routing: signal-led, entropy-aware, ruthlessly clear.
                </Translate>
              </span>
            </p>
          </div>
        </div>
      </div>
    </header>
  )
}

function CapabilitySection(): JSX.Element {
  return (
    <section className={styles.capabilitySection}>
      <div className="site-shell-container">
        <div className={styles.sectionHeading}>
          <SectionLabel>
            <Translate id="homepage.capabilities.label">Core logic</Translate>
          </SectionLabel>
          <div>
            <h2>
              <Translate id="homepage.capabilities.heading">
                Charting the LLM system brain.
              </Translate>
            </h2>
            <p>
              <Translate id="homepage.capabilities.copy">
                A research-driven stack for uncharted territory, probing the frontier where signals,
                projections, policies, and models converge into one intelligence layer.
              </Translate>
            </p>
          </div>
        </div>

        <div className={styles.capabilityGrid}>
          {capabilityCards.map(card => (
            <article key={card.title} className={styles.capabilityCard}>
              <div className={styles.capabilityCardHead}>
                <SectionLabel>{card.title}</SectionLabel>
                <div className={styles.capabilityVisual}>
                  <CapabilityGlyph kind={card.kind} className={styles.capabilityGlyph} />
                </div>
              </div>
              <p>{card.text}</p>
            </article>
          ))}
        </div>
      </div>
    </section>
  )
}

function EncoderIntelligenceSection(): JSX.Element {
  return (
    <section className={styles.encoderSection}>
      <div className="site-shell-container">
        <div className={styles.sectionHeading}>
          <SectionLabel>
            <Translate id="homepage.aiTech.label">Built on Encoder Models</Translate>
          </SectionLabel>
          <div>
            <h2>
              <Translate id="homepage.aiTech.title">Encoder-Based Intelligence</Translate>
            </h2>
            <p>
              <Translate id="homepage.aiTech.description">
                Purpose-built encoders read intent, rank relevance, and classify modality before
                generation begins.
              </Translate>
            </p>
          </div>
        </div>

        <div className={styles.encoderShowcase}>
          <div className={styles.encoderLeadStack}>
            <div className={styles.encoderLead}>
              <div className={styles.encoderLeadCopy}>
                <SectionLabel>
                  <Translate id="homepage.aiTech.leadLabel">Signal surfaces</Translate>
                </SectionLabel>
                <p>
                  <Translate id="homepage.aiTech.leadCopy">
                    Sequence classification, token labeling, embeddings, and reranking collapse into
                    one system-intelligence layer.
                  </Translate>
                </p>
              </div>

              <div className={styles.encoderTrackList}>
                {encoderTracks.map(track => (
                  <div key={track.label} className={styles.encoderTrack}>
                    <span className={styles.encoderTrackLabel}>{track.label}</span>
                    <span>{track.text}</span>
                  </div>
                ))}
              </div>

              <div className={styles.encoderActions}>
                <PillLink
                  href="https://huggingface.co/LLM-Semantic-Router"
                  rel="noreferrer"
                  target="_blank"
                >
                  <Translate id="homepage.aiTech.primaryCta">Hugging Face Models</Translate>
                </PillLink>
              </div>
            </div>

            <article className={`${styles.encoderCard} ${styles.encoderSpotlightCard}`}>
              <span className={styles.encoderCardMarker}>{encoderSpotlightCard.marker}</span>
              <div className={styles.encoderCardCopy}>
                <h3>{encoderSpotlightCard.title}</h3>
                <p>{encoderSpotlightCard.text}</p>
              </div>
            </article>
          </div>

          <div className={styles.encoderPipelineFrame}>
            <TransformerPipelineAnimation />
          </div>
        </div>

        <div className={styles.encoderCardGrid}>
          {encoderCards.map(card => (
            <article key={card.marker} className={styles.encoderCard}>
              <span className={styles.encoderCardMarker}>{card.marker}</span>
              <div className={styles.encoderCardCopy}>
                <h3>{card.title}</h3>
                <p>{card.text}</p>
              </div>
            </article>
          ))}
        </div>
      </div>
    </section>
  )
}

function ClosingBands(): JSX.Element {
  return (
    <section className={styles.closingBands}>
      <div className="site-shell-container">
        <div className={styles.bandGrid}>
          <div className={styles.band}>
            <SectionLabel>
              <Translate id="homepage.band.docs.label">Documentation</Translate>
            </SectionLabel>
            <h3>
              <Translate id="homepage.band.docs.title">Architecture, written to be used.</Translate>
            </h3>
            <p>
              <Translate id="homepage.band.docs.text">
                Install, configure, train, and operate from one dense documentation graph.
              </Translate>
            </p>
            <PillLink to="/docs/intro">Docs index</PillLink>
          </div>

          <div className={styles.band}>
            <SectionLabel>
              <Translate id="homepage.band.community.label">Community</Translate>
            </SectionLabel>
            <h3>
              <Translate id="homepage.band.community.title">Research and builders in one loop.</Translate>
            </h3>
            <p>
              <Translate id="homepage.band.community.text">
                Papers, working groups, and contributors evolve the same system in public.
              </Translate>
            </p>
            <PillLink to="/community/team" muted>Community routes</PillLink>
          </div>
        </div>
      </div>
    </section>
  )
}

export default function Home(): JSX.Element {
  const { siteConfig } = useDocusaurusContext()

  return (
    <Layout
      title={siteConfig.title}
      description={translate({
        id: 'homepage.meta.description',
        message: 'Signal-driven decision routing for mixture-of-model serving.',
      })}
    >
      <main className={styles.page}>
        <DitherHero />

        <section className={styles.statsSection}>
          <div className="site-shell-container">
            <StatStrip items={heroStats} />
          </div>
        </section>

        <InstallQuickStartSection />
        <ResearchPaperCarousel />
        <CapabilitySection />
        <PaperFigureShowcase />
        <EncoderIntelligenceSection />
        <TeamCarousel />
        <AcknowledgementsSection />
        <ClosingBands />
      </main>
    </Layout>
  )
}
