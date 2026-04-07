import React from 'react'
import Head from '@docusaurus/Head'
import Layout from '@theme/Layout'
import Translate, { translate } from '@docusaurus/Translate'
import useDocusaurusContext from '@docusaurus/useDocusaurusContext'
import AcknowledgementsSection from '@site/src/components/AcknowledgementsSection'
import InstallQuickStartSection from '@site/src/components/InstallQuickStartSection'
import PaperFigureShowcase from '@site/src/components/PaperFigureShowcase'
import ResearchPaperCarousel from '@site/src/components/ResearchPaperCarousel'
import TeamCarousel from '@site/src/components/TeamCarousel'
import { researchPapers } from '@site/src/data/researchContent'
import { SITE_SOCIAL_PREVIEW_IMAGE_PATH } from '@site/src/data/socialPreview'
import TransformerPipelineAnimation from '@site/src/components/TransformerPipelineAnimation'
import CapabilityGlyph, { type CapabilityGlyphKind } from '@site/src/components/site/CapabilityGlyph'
import DitherField from '@site/src/components/site/DitherField'
import { PageIntro, PillLink, SectionLabel, StatStrip } from '@site/src/components/site/Chrome'
import styles from './index.module.css'

const paperCount = researchPapers.length
const homepageMetaTitle = translate({
  id: 'homepage.meta.title',
  message: 'LLM Routing Control Plane for Mixture-of-Models',
})
const homepageMetaDescription = translate({
  id: 'homepage.meta.description',
  message: 'Signal-driven LLM routing and model selection for mixture-of-model serving, AI gateways, and policy-aware control planes.',
})
const homepageSocialTitle = translate({
  id: 'homepage.meta.socialTitle',
  message: 'vLLM Semantic Router — Signal-Driven LLM Routing',
})

const heroStats = [
  {
    label: translate({ id: 'homepage.stats.signals.label', message: 'Signals' }),
    value: '16',
    description: translate({
      id: 'homepage.stats.signals.description',
      message: '16 signal families across 5 heuristic and 11 learned detectors, including knowledge base routing and history-aware reasks.',
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

interface ValueCard {
  detail: string
  index: string
  kind: CapabilityGlyphKind
  text: string
  title: string
}

const problemAxes = [
  translate({ id: 'homepage.capabilities.axis.capability', message: 'Capability' }),
  translate({ id: 'homepage.capabilities.axis.cost', message: 'Cost' }),
  translate({ id: 'homepage.capabilities.axis.privacy', message: 'Privacy' }),
  translate({ id: 'homepage.capabilities.axis.latency', message: 'Latency' }),
]

const problemTasks = [
  translate({
    id: 'homepage.capabilities.task.choose',
    message: 'Choose the right model lane for each request.',
  }),
  translate({
    id: 'homepage.capabilities.task.connect',
    message: 'Connect edge, private, and frontier models without fragmenting the product.',
  }),
  translate({
    id: 'homepage.capabilities.task.govern',
    message: 'Govern cost, safety, and privacy at routing time.',
  }),
]

const problemMeta = [
  translate({ id: 'homepage.capabilities.meta.selection', message: 'Selection' }),
  translate({ id: 'homepage.capabilities.meta.connection', message: 'Connection' }),
  translate({ id: 'homepage.capabilities.meta.governance', message: 'Governance' }),
]

const valueCards: ValueCard[] = [
  {
    index: '01',
    kind: 'economics',
    title: translate({ id: 'homepage.capabilities.value1.title', message: 'Token economics' }),
    text: translate({
      id: 'homepage.capabilities.value1.text',
      message: 'Send routine traffic to efficient lanes, reserve frontier reasoning for the moments that pay back, and turn model choice into compounding ROI.',
    }),
    detail: translate({
      id: 'homepage.capabilities.value1.detail',
      message: 'More useful output per token.',
    }),
  },
  {
    index: '02',
    kind: 'safety',
    title: translate({ id: 'homepage.capabilities.value2.title', message: 'LLM safety' }),
    text: translate({
      id: 'homepage.capabilities.value2.text',
      message: 'Move jailbreak, PII, and hallucination handling into the decision path so unsafe traffic is intercepted before it becomes product behavior.',
    }),
    detail: translate({
      id: 'homepage.capabilities.value2.detail',
      message: 'Safety becomes a control layer, not a patch.',
    }),
  },
  {
    index: '03',
    kind: 'mesh',
    title: translate({ id: 'homepage.capabilities.value3.title', message: 'Fullmesh intelligence' }),
    text: translate({
      id: 'homepage.capabilities.value3.text',
      message: 'Coordinate local, private, and frontier models as one adaptive mesh for personal AI at the edge and intelligent MaaS in the cloud.',
    }),
    detail: translate({
      id: 'homepage.capabilities.value3.detail',
      message: 'One architecture from device to managed service.',
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
      message: 'Embedding and rerank paths for semantic cache, knowledge base routing, reask similarity scoring, and candidate ranking.',
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
        <div className={styles.capabilityFrame}>
          <div className={styles.problemPanel}>
            <div className={styles.problemIntro}>
              <SectionLabel>
                <Translate id="homepage.capabilities.label">Why this matters</Translate>
              </SectionLabel>
              <h2>
                <Translate id="homepage.capabilities.heading">
                  Too many models. Too many boundaries.
                </Translate>
              </h2>
              <p>
                <Translate id="homepage.capabilities.copy">
                  Models now vary across capability, cost, privacy, and latency. Once you operate more
                  than one lane, the hard part is no longer prompting a single model. It is selecting,
                  connecting, and governing the right model system for every request.
                </Translate>
              </p>
              <div className={styles.problemAxes}>
                {problemAxes.map(axis => (
                  <span key={axis} className={styles.problemAxis}>
                    {axis}
                  </span>
                ))}
              </div>
            </div>

            <aside className={styles.problemChecklist}>
              <div className={styles.problemChecklistHeader}>
                <SectionLabel>
                  <Translate id="homepage.capabilities.checklist.label">The system job</Translate>
                </SectionLabel>
                <p>
                  <Translate id="homepage.capabilities.checklist.copy">
                    Every request crosses the same operating questions before it becomes product behavior.
                  </Translate>
                </p>
              </div>
              <ul className={styles.problemChecklistList}>
                {problemTasks.map(task => (
                  <li key={task} className={styles.problemChecklistItem}>
                    {task}
                  </li>
                ))}
              </ul>
              <div className={styles.problemChecklistFooter}>
                <p>
                  <Translate id="homepage.capabilities.checklist.footer">
                    Selection, connection, and governance have to move together.
                  </Translate>
                </p>
                <div className={styles.problemChecklistMeta}>
                  {problemMeta.map(item => (
                    <span key={item} className={styles.problemChecklistMetaItem}>
                      {item}
                    </span>
                  ))}
                </div>
              </div>
            </aside>
          </div>

          <div className={styles.valueIntro}>
            <SectionLabel>
              <Translate id="homepage.capabilities.values.label">What vLLM Semantic Router enables</Translate>
            </SectionLabel>
            <p>
              <Translate id="homepage.capabilities.values.copy">
                Routing becomes the control plane.
              </Translate>
            </p>
          </div>

          <div className={styles.valueGrid}>
            {valueCards.map(card => (
              <article key={card.title} className={styles.valueCard}>
                <div className={styles.valueCardHeader}>
                  <span className={styles.valueCardIndex}>{card.index}</span>
                  <div className={styles.valueGlyphShell}>
                    <CapabilityGlyph kind={card.kind} className={styles.valueGlyph} />
                  </div>
                </div>
                <div className={styles.valueCardCopy}>
                  <h3>{card.title}</h3>
                  <p>{card.text}</p>
                </div>
                <p className={styles.valueCardDetail}>{card.detail}</p>
              </article>
            ))}
          </div>
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
            <PillLink to="/docs/intro">
              <Translate id="homepage.band.docs.cta">Docs index</Translate>
            </PillLink>
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
            <PillLink to="/community/team" muted>
              <Translate id="homepage.band.community.cta">Community routes</Translate>
            </PillLink>
          </div>
        </div>
      </div>
    </section>
  )
}

export default function Home(): JSX.Element {
  const { siteConfig } = useDocusaurusContext()
  const ogImage = new URL(SITE_SOCIAL_PREVIEW_IMAGE_PATH, siteConfig.url).toString()
  const homepageStructuredData = {
    '@context': 'https://schema.org',
    '@type': 'WebSite',
    'name': 'vLLM Semantic Router',
    'url': siteConfig.url,
    'description': homepageMetaDescription,
    'inLanguage': ['en-US', 'zh-Hans'],
    'publisher': {
      '@type': 'Organization',
      'name': 'vLLM Semantic Router Team',
      'url': 'https://github.com/vllm-project/semantic-router',
    },
    'sameAs': [
      'https://github.com/vllm-project/semantic-router',
      'https://huggingface.co/LLM-Semantic-Router',
    ],
  }

  return (
    <Layout
      title={homepageMetaTitle}
      description={homepageMetaDescription}
    >
      <Head>
        <meta property="og:title" content={homepageSocialTitle} />
        <meta property="og:description" content={homepageMetaDescription} />
        <meta property="og:image" content={ogImage} />
        <meta property="og:image:alt" content="vLLM Semantic Router social preview" />
        <meta property="og:type" content="website" />
        <meta name="twitter:card" content="summary_large_image" />
        <meta name="twitter:title" content={homepageSocialTitle} />
        <meta name="twitter:description" content={homepageMetaDescription} />
        <meta name="twitter:image" content={ogImage} />
        <meta name="twitter:image:alt" content="vLLM Semantic Router social preview" />
        <script
          type="application/ld+json"
          dangerouslySetInnerHTML={{ __html: JSON.stringify(homepageStructuredData) }}
        />
      </Head>
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
