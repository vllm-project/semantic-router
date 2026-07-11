import React from 'react'
import Head from '@docusaurus/Head'
import Layout from '@theme/Layout'
import Translate, { translate } from '@docusaurus/Translate'
import useBaseUrl from '@docusaurus/useBaseUrl'
import useDocusaurusContext from '@docusaurus/useDocusaurusContext'
import Claude from '@lobehub/icons/es/Claude/components/Mono'
import DeepSeek from '@lobehub/icons/es/DeepSeek/components/Mono'
import Gemini from '@lobehub/icons/es/Gemini/components/Mono'
import Grok from '@lobehub/icons/es/Grok/components/Mono'
import Kimi from '@lobehub/icons/es/Kimi/components/Mono'
import Meta from '@lobehub/icons/es/Meta/components/Mono'
import Minimax from '@lobehub/icons/es/Minimax/components/Mono'
import Mistral from '@lobehub/icons/es/Mistral/components/Mono'
import OpenAI from '@lobehub/icons/es/OpenAI/components/Mono'
import Qwen from '@lobehub/icons/es/Qwen/components/Mono'
import Zhipu from '@lobehub/icons/es/Zhipu/components/Mono'
import AcknowledgementsSection from '@site/src/components/AcknowledgementsSection'
import InstallQuickStartSection from '@site/src/components/InstallQuickStartSection'
import PaperFigureShowcase from '@site/src/components/PaperFigureShowcase'
import ResearchPaperCarousel from '@site/src/components/ResearchPaperCarousel'
import TeamCarousel from '@site/src/components/TeamCarousel'
import { researchPapers } from '@site/src/data/researchContent'
import { SITE_SOCIAL_PREVIEW_IMAGE_PATH } from '@site/src/data/socialPreview'
import TransformerPipelineAnimation from '@site/src/components/TransformerPipelineAnimation'
import DitherField from '@site/src/components/site/DitherField'
import { PageIntro, PillLink, SectionLabel, StatStrip } from '@site/src/components/site/Chrome'
import styles from './index.module.css'

const paperCount = researchPapers.length
const homepageMetaTitle = translate({
  id: 'homepage.meta.title',
  message: 'Open-Source Runtime for Mixture-of-Models',
})
const homepageMetaDescription = translate({
  id: 'homepage.meta.description',
  message: 'Mixture-of-Models is a next-generation model architecture shaped by preference. vLLM Semantic Router is its open-source runtime.',
})
const homepageSocialTitle = translate({
  id: 'homepage.meta.socialTitle',
  message: 'vLLM Semantic Router | Runtime for Mixture-of-Models',
})

type HeroModelLogo = {
  label: string
  Icon: React.ElementType
}

const heroModelLogos: HeroModelLogo[] = [
  { label: 'Kimi', Icon: Kimi },
  { label: 'Zhipu', Icon: Zhipu },
  { label: 'MiniMax', Icon: Minimax },
  { label: 'ChatGPT', Icon: OpenAI },
  { label: 'Claude', Icon: Claude },
  { label: 'Gemini', Icon: Gemini },
  { label: 'DeepSeek', Icon: DeepSeek },
  { label: 'Qwen', Icon: Qwen },
  { label: 'Llama', Icon: Meta },
  { label: 'Mistral', Icon: Mistral },
  { label: 'Grok', Icon: Grok },
]

const heroStats = [
  {
    label: translate({ id: 'homepage.stats.signals.label', message: 'Signals' }),
    value: '16',
    description: translate({
      id: 'homepage.stats.signals.description',
      message: '16 signal families across heuristic and learned detectors, from knowledge base routing to history-aware reasks.',
    }),
  },
  {
    label: translate({ id: 'homepage.stats.algorithms.label', message: 'Selection' }),
    value: '12',
    description: translate({
      id: 'homepage.stats.algorithms.description',
      message: '12 routing strategies spanning rules, latency heuristics, reinforcement learning, and ML selection.',
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

interface FragmentationRow {
  dimension: string
  id: 'models' | 'compute' | 'location' | 'preference'
  reality: string
  value: string
}

const comparisonHeaders = {
  dimension: translate({ id: 'homepage.capabilities.table.dimension', message: 'Dimension' }),
  reality: translate({ id: 'homepage.capabilities.table.reality', message: 'Fragmented today' }),
  value: translate({ id: 'homepage.capabilities.table.value', message: 'With vLLM SR' }),
}

const fragmentationRows: FragmentationRow[] = [
  {
    dimension: translate({ id: 'homepage.capabilities.axis.models', message: 'Models' }),
    id: 'models',
    reality: translate({
      id: 'homepage.capabilities.models.reality',
      message: 'Closed frontier, open, domain-specialized, and edge models excel at different work.',
    }),
    value: translate({
      id: 'homepage.capabilities.models.value',
      message: 'Route one—or coordinate many.',
    }),
  },
  {
    dimension: translate({ id: 'homepage.capabilities.axis.compute', message: 'Compute' }),
    id: 'compute',
    reality: translate({
      id: 'homepage.capabilities.compute.reality',
      message: 'New and legacy GPUs, accelerators, edge devices, and cloud capacity coexist.',
    }),
    value: translate({
      id: 'homepage.capabilities.compute.value',
      message: 'Route across configured model endpoints on heterogeneous compute.',
    }),
  },
  {
    dimension: translate({ id: 'homepage.capabilities.axis.location', message: 'Location' }),
    id: 'location',
    reality: translate({
      id: 'homepage.capabilities.location.reality',
      message: 'Inference spans edge, private data centers, and cloud.',
    }),
    value: translate({
      id: 'homepage.capabilities.location.value',
      message: 'Route across locations within privacy and data boundaries.',
    }),
  },
  {
    dimension: translate({ id: 'homepage.capabilities.axis.preference', message: 'Preference' }),
    id: 'preference',
    reality: translate({
      id: 'homepage.capabilities.preference.reality',
      message: '“Best” varies by user, product, and workload.',
    }),
    value: translate({
      id: 'homepage.capabilities.preference.value',
      message: 'Let preference shape every model path.',
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
  const marqueeCopies = [0, 1]
  const marqueeRepeats = [0, 1, 2]
  const heroLogoSrc = useBaseUrl('/img/artworks/vllm-sr-logo.dark.svg')
  const heroLogoAlt = translate({
    id: 'homepage.hero.logoAlt',
    message: 'vLLM Semantic Router logo',
  })

  return (
    <section className={styles.heroStage}>
      <header className={styles.hero}>
        <DitherField className={styles.heroNoise} />
        <div className="site-shell-container">
          <div className={styles.heroGrid}>
            <div className={styles.heroIntro}>
              <div className={styles.heroBrandLockup}>
                <img
                  src={heroLogoSrc}
                  alt={heroLogoAlt}
                  className={styles.heroBrandLogo}
                  decoding="async"
                  loading="eager"
                />
              </div>

              <PageIntro
                align="center"
                className={styles.heroIntroPanel}
                label={<Translate id="homepage.hero.label">Mixture-of-Models</Translate>}
                title={(
                  <span className={styles.heroTitle}>
                    <span className={`${styles.heroTitleLine} ${styles.heroTitleAccent}`}>
                      <Translate id="homepage.hero.line1">Intelligence,</Translate>
                    </span>
                    <span className={styles.heroTitleLine}>
                      <Translate id="homepage.hero.line2">composed for you.</Translate>
                    </span>
                  </span>
                )}
                description={(
                  <span className={styles.heroDescriptionText}>
                    <Translate id="homepage.hero.description">
                      One model or many, across compute and locations—shaped by your priorities.
                    </Translate>
                  </span>
                )}
                actions={(
                  <>
                    <PillLink
                      className={styles.heroPrimaryCta}
                      href="https://play.vllm-semantic-router.com/"
                      rel="noreferrer"
                      target="_blank"
                    >
                      <Translate id="homepage.hero.primaryCta">Try the Playground</Translate>
                    </PillLink>
                    <PillLink className={styles.heroSecondaryCta} to="/docs/intro" muted>
                      <Translate id="homepage.hero.secondaryCta">Explore the Docs</Translate>
                    </PillLink>
                  </>
                )}
              />
            </div>
          </div>
        </div>
      </header>

      <section
        className={styles.heroModelSection}
        aria-label={translate({
          id: 'homepage.hero.modelBand.aria',
          message: 'Mixture-of-Models ecosystem',
        })}
      >
        <div className={styles.heroModelBand}>
          <div className={styles.heroModelBandHeader}>
            <span className={styles.heroModelBandEyebrow}>
              <Translate id="homepage.hero.modelBand.eyebrow">Mixture-of-Models</Translate>
            </span>
          </div>

          <div className={styles.heroModelBandViewport} aria-hidden="true">
            <div className={styles.heroModelBandTrack}>
              {marqueeCopies.map(copyIndex => (
                <div
                  key={`hero-model-sequence-${copyIndex}`}
                  className={styles.heroModelBandSequence}
                >
                  {marqueeRepeats.map(repeatIndex =>
                    heroModelLogos.map(({ label, Icon }) => (
                      <div key={`${copyIndex}-${repeatIndex}-${label}`} className={styles.heroModelChip}>
                        <span className={styles.heroModelChipIcon} aria-hidden="true">
                          <Icon size={20} className={styles.heroModelGlyph} />
                        </span>
                        <span className={styles.heroModelChipLabel}>{label}</span>
                      </div>
                    )),
                  )}
                </div>
              ))}
            </div>
          </div>
        </div>
      </section>
    </section>
  )
}

function CapabilitySection(): JSX.Element {
  return (
    <section className={styles.capabilitySection}>
      <div className="site-shell-container">
        <div className={styles.capabilityFrame}>
          <div className={styles.comparisonHeading}>
            <SectionLabel>
              <Translate id="homepage.capabilities.label">Why Mixture-of-Models</Translate>
            </SectionLabel>
            <h2 id="fragmentation-comparison-title">
              <Translate id="homepage.capabilities.heading">
                Fragmented by nature. Composed by preference.
              </Translate>
            </h2>
          </div>

          <div className={styles.comparisonTableWrap}>
            <table className={styles.comparisonTable} aria-labelledby="fragmentation-comparison-title">
              <thead>
                <tr>
                  <th scope="col">{comparisonHeaders.dimension}</th>
                  <th scope="col">{comparisonHeaders.reality}</th>
                  <th scope="col">{comparisonHeaders.value}</th>
                </tr>
              </thead>
              <tbody>
                {fragmentationRows.map(row => (
                  <tr key={row.id}>
                    <th scope="row" className={styles.comparisonDimension}>{row.dimension}</th>
                    <td data-label={comparisonHeaders.reality}>{row.reality}</td>
                    <td data-label={comparisonHeaders.value} className={styles.comparisonValue}>{row.value}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className={styles.comparisonCta}>
            <PillLink to="/docs/intro" muted>
              <Translate id="homepage.capabilities.docsCta">Explore how it works</Translate>
            </PillLink>
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
        <meta
          name="keywords"
          content="Mixture-of-Models runtime, preference-driven AI, open-source LLM router, multi-model routing, model orchestration, model selection, model cascade, Fusion API, micro-agent workflows, semantic router, policy-aware routing, vLLM"
        />
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

        <CapabilitySection />

        <section className={styles.statsSection}>
          <div className="site-shell-container">
            <StatStrip items={heroStats} />
          </div>
        </section>

        <InstallQuickStartSection />
        <ResearchPaperCarousel />
        <PaperFigureShowcase />
        <EncoderIntelligenceSection />
        <TeamCarousel />
        <AcknowledgementsSection />
        <ClosingBands />
      </main>
    </Layout>
  )
}
