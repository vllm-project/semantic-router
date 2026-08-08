import React from 'react'
import Head from '@docusaurus/Head'
import Layout from '@theme/Layout'
import Translate, { translate } from '@docusaurus/Translate'
import useDocusaurusContext from '@docusaurus/useDocusaurusContext'
import ValuePillars from '@site/src/components/homepage/ValuePillars'
import IntegrationArchitecture from '@site/src/components/homepage/IntegrationArchitecture'
import UseCaseExplorer from '@site/src/components/homepage/UseCaseExplorer'
import CompatibilityBand from '@site/src/components/homepage/CompatibilityBand'
import InstallQuickStartSection from '@site/src/components/InstallQuickStartSection'
import PaperFigureShowcase from '@site/src/components/PaperFigureShowcase'
import ResearchPaperCarousel from '@site/src/components/ResearchPaperCarousel'
import TeamCarousel from '@site/src/components/TeamCarousel'
import TestimonialsRail from '@site/src/components/TestimonialsRail'
import SponsorsSection from '@site/src/components/homepage/SponsorsSection'
import { researchPapers } from '@site/src/data/researchContent'
import { SITE_SOCIAL_PREVIEW_IMAGE_PATH } from '@site/src/data/socialPreview'
import SemanticTerrainHero from '@site/src/components/site/SemanticTerrainHero'
import ScrollReveal from '@site/src/components/site/ScrollReveal'
import {
  PillLink,
  SectionLabel,
  StatStrip,
} from '@site/src/components/site/Chrome'
import styles from './index.module.css'

const paperCount = researchPapers.length
const homepageMetaTitle = translate({
  id: 'homepage.meta.title',
  message: 'Build Your Mixture-of-Models',
})
const homepageMetaDescription = translate({
  id: 'homepage.meta.description',
  message:
    'We believe Mixture-of-Models is the next-generation model architecture for heterogeneous LLM inference. vLLM Semantic Router makes it executable.',
})
const homepageSocialTitle = translate({
  id: 'homepage.meta.socialTitle',
  message: 'Build Your Mixture-of-Models | vLLM Semantic Router',
})

const heroStats = [
  {
    label: translate({
      id: 'homepage.stats.signals.label',
      message: 'Signals',
    }),
    value: '16',
    description: translate({
      id: 'homepage.stats.signals.description',
      message:
        '16 signal families across heuristic and learned detectors, from knowledge base routing to history-aware reasks.',
    }),
  },
  {
    label: translate({
      id: 'homepage.stats.algorithms.label',
      message: 'Selection',
    }),
    value: '12',
    description: translate({
      id: 'homepage.stats.algorithms.description',
      message:
        '12 routing strategies spanning rules, latency heuristics, reinforcement learning, and ML selection.',
    }),
  },
  {
    label: translate({ id: 'homepage.stats.papers.label', message: 'Papers' }),
    value: String(paperCount).padStart(2, '0'),
    description: translate(
      {
        id: 'homepage.stats.papers.description',
        message:
          '{count} research papers spanning routing, systems, safety, and multimodality.',
      },
      { count: paperCount },
    ),
  },
]

const architectureDimensions = [
  {
    marker: '01',
    dimension: translate({
      id: 'homepage.capabilities.axis.models',
      message: 'Models',
    }),
    fragmented: translate({
      id: 'homepage.capabilities.models.reality',
      message: 'Models specialize in different work.',
    }),
    unified: translate({
      id: 'homepage.capabilities.models.value',
      message: 'Compose personalized model paths.',
    }),
  },
  {
    marker: '02',
    dimension: translate({
      id: 'homepage.capabilities.axis.compute',
      message: 'Compute',
    }),
    fragmented: translate({
      id: 'homepage.capabilities.compute.reality',
      message: 'GPUs, accelerators, edge, and cloud coexist.',
    }),
    unified: translate({
      id: 'homepage.capabilities.compute.value',
      message: 'Route across heterogeneous compute.',
    }),
  },
  {
    marker: '03',
    dimension: translate({
      id: 'homepage.capabilities.axis.location',
      message: 'Location',
    }),
    fragmented: translate({
      id: 'homepage.capabilities.location.reality',
      message: 'Inference spans edge, private, and cloud.',
    }),
    unified: translate({
      id: 'homepage.capabilities.location.value',
      message: 'Keep data within its boundaries.',
    }),
  },
  {
    marker: '04',
    dimension: translate({
      id: 'homepage.capabilities.axis.preference',
      message: 'Preference',
    }),
    fragmented: translate({
      id: 'homepage.capabilities.preference.reality',
      message: '“Best” changes by user and workload.',
    }),
    unified: translate({
      id: 'homepage.capabilities.preference.value',
      message: 'Make every preference executable.',
    }),
  },
]

const momScorecards = [
  {
    title: translate({
      id: 'homepage.momProof.livecodebench.title',
      message: 'LiveCodeBench',
    }),
    result: translate({
      id: 'homepage.momProof.livecodebench.result',
      message: '92.6 vs Fugu Ultra 92.0',
    }),
    image: '/img/mom-proof/livecodebench-scorecard-dark.png',
    alt: translate({
      id: 'homepage.momProof.livecodebench.alt',
      message: 'LiveCodeBench dark scorecard showing VSR Closed at 92.6',
    }),
  },
  {
    title: translate({
      id: 'homepage.momProof.gpqa.title',
      message: 'GPQA-Diamond',
    }),
    result: translate({
      id: 'homepage.momProof.gpqa.result',
      message: '96.0 vs Fugu Ultra 95.5',
    }),
    image: '/img/mom-proof/gpqa-diamond-scorecard-dark.png',
    alt: translate({
      id: 'homepage.momProof.gpqa.alt',
      message: 'GPQA-Diamond dark scorecard showing VSR Closed at 96.0',
    }),
  },
  {
    title: translate({
      id: 'homepage.momProof.hle.title',
      message: 'Humanity\'s Last Exam',
    }),
    result: translate({
      id: 'homepage.momProof.hle.result',
      message: '50.0 matches Fugu Ultra',
    }),
    image: '/img/mom-proof/humanitys-last-exam-scorecard-dark.png',
    alt: translate({
      id: 'homepage.momProof.hle.alt',
      message: 'Humanity\'s Last Exam dark scorecard showing VSR Closed at 50.0',
    }),
  },
]

function CapabilitySection(): JSX.Element {
  return (
    <section
      className={styles.capabilitySection}
      aria-labelledby="mixture-architecture-title"
    >
      <div className="site-shell-container">
        <ScrollReveal>
          <div className={styles.capabilityFrame}>
            <header className={styles.capabilityHeading}>
              <SectionLabel className={styles.capabilityLabel}>
                <Translate id="homepage.capabilities.label">Architecture</Translate>
              </SectionLabel>
              <h2 id="mixture-architecture-title">
                <Translate id="homepage.capabilities.heading">
                  Unify heterogeneous inference.
                </Translate>
              </h2>
            </header>

            <div className={styles.capabilitySummary}>
              <p>
                <Translate id="homepage.capabilities.description">
                  Unify a fragmented model landscape across four dimensions.
                </Translate>
              </p>
              <PillLink className={styles.capabilityCta} to="/docs/intro">
                <Translate id="homepage.capabilities.docsCta">
                  Explore how it works
                </Translate>
              </PillLink>
            </div>

            <div
              className={styles.architectureMatrix}
              role="table"
              aria-label={translate({
                id: 'homepage.capabilities.table.aria',
                message: 'Fragmented inference compared with vLLM Semantic Router',
              })}
            >
              <div className={styles.matrixHeader} role="row">
                <span role="columnheader">
                  <Translate id="homepage.capabilities.table.dimension">
                    Dimension
                  </Translate>
                </span>
                <span role="columnheader">
                  <Translate id="homepage.capabilities.table.reality">
                    Fragmented today
                  </Translate>
                </span>
                <span role="columnheader">
                  <Translate id="homepage.capabilities.table.value">
                    With vLLM SR
                  </Translate>
                </span>
              </div>

              {architectureDimensions.map(item => (
                <div key={item.marker} className={styles.matrixRow} role="row">
                  <div className={styles.matrixDimension} role="rowheader">
                    <span aria-hidden="true">{item.marker}</span>
                    <strong>{item.dimension}</strong>
                  </div>
                  <div className={styles.matrixFragmented} role="cell">
                    <span className={styles.matrixMobileLabel}>
                      <Translate id="homepage.capabilities.table.reality">
                        Fragmented today
                      </Translate>
                    </span>
                    <p>{item.fragmented}</p>
                  </div>
                  <div className={styles.matrixUnified} role="cell">
                    <span className={styles.matrixMobileLabel}>
                      <Translate id="homepage.capabilities.table.value">
                        With vLLM SR
                      </Translate>
                    </span>
                    <p>{item.unified}</p>
                  </div>
                </div>
              ))}
            </div>

            <div className={styles.capabilityStats}>
              <StatStrip items={heroStats} />
            </div>
          </div>
        </ScrollReveal>
      </div>
    </section>
  )
}

function MixtureOfModelsProofSection(): JSX.Element {
  return (
    <section className={styles.momProofSection} aria-labelledby="mom-proof-title">
      <div className="site-shell-container">
        <ScrollReveal>
          <div className={styles.momProofHeading}>
            <SectionLabel>
              <Translate id="homepage.momProof.label">
                Mixture-of-Models proof
              </Translate>
            </SectionLabel>
            <div>
              <h2 id="mom-proof-title">
                <Translate id="homepage.momProof.title">
                  One model API can beat frontier models.
                </Translate>
              </h2>
              <p>
                <Translate id="homepage.momProof.description">
                  vLLM Semantic Router keeps the public surface as vllm-sr/auto,
                  then coordinates closed, open, and hybrid model pools inside the
                  serving layer.
                </Translate>
              </p>
            </div>
          </div>
        </ScrollReveal>

        <ScrollReveal delay={70}>
          <div className={styles.momProofFrame}>
            <div className={styles.momProofArchitecture}>
              <div className={styles.momProofArchitectureCopy}>
                <SectionLabel>
                  <Translate id="homepage.momProof.architectureLabel">
                    Router-side collaboration
                  </Translate>
                </SectionLabel>
                <h3>
                  <Translate id="homepage.momProof.architectureTitle">
                    The app calls one model. The router builds the team.
                  </Translate>
                </h3>
                <p>
                  <Translate id="homepage.momProof.architectureCopy">
                    Route by task shape, risk, confidence, and model capability;
                    run bounded collaboration; return one OpenAI-compatible
                    response.
                  </Translate>
                </p>
              </div>

              <div className={styles.momProofArchitectureImageWrap}>
                <img
                  className={styles.momProofArchitectureImage}
                  src="/img/mom-proof/architecture-router-dark.png"
                  alt={translate({
                    id: 'homepage.momProof.architectureAlt',
                    message:
                      'vLLM Semantic Router routes heterogeneous closed and open model pools',
                  })}
                  loading="lazy"
                />
              </div>
            </div>
          </div>
        </ScrollReveal>

        <ScrollReveal delay={120}>
          <div className={styles.momScorecardGrid}>
            {momScorecards.map(card => (
              <article key={card.image} className={styles.momScorecard}>
                <div className={styles.momScorecardHeader}>
                  <h3>{card.title}</h3>
                  <p>{card.result}</p>
                </div>
                <img
                  className={styles.momScorecardImage}
                  src={card.image}
                  alt={card.alt}
                  loading="lazy"
                />
              </article>
            ))}
          </div>
        </ScrollReveal>
      </div>
    </section>
  )
}

function FinalCtaSection(): JSX.Element {
  return (
    <section className={styles.finalCtaSection}>
      <div className="site-shell-container">
        <ScrollReveal>
          <div className={styles.finalCtaFrame}>
            <div className={styles.finalCtaCopy}>
              <SectionLabel>
                <Translate id="homepage.finalCta.label">Start building</Translate>
              </SectionLabel>
              <h2>
                <Translate id="homepage.finalCta.title">
                  Compose your Mixture-of-Models.
                </Translate>
              </h2>
              <p>
                <Translate id="homepage.finalCta.description">
                  Shape every model path with signals, preferences, and policy.
                </Translate>
              </p>
            </div>
            <div className={styles.finalCtaActions}>
              <PillLink
                href="https://app.vllm-sr.ai/playground"
                rel="noreferrer"
                target="_blank"
              >
                <Translate id="homepage.finalCta.playground">Try the Playground</Translate>
              </PillLink>
              <PillLink to="/docs/intro" muted>
                <Translate id="homepage.finalCta.docs">Explore the Docs</Translate>
              </PillLink>
            </div>
          </div>
        </ScrollReveal>
      </div>
    </section>
  )
}

export default function Home(): JSX.Element {
  const { siteConfig } = useDocusaurusContext()
  const ogImage = new URL(
    SITE_SOCIAL_PREVIEW_IMAGE_PATH,
    siteConfig.url,
  ).toString()
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
    <Layout title={homepageMetaTitle} description={homepageMetaDescription}>
      <Head>
        <meta property="og:title" content={homepageSocialTitle} />
        <meta property="og:description" content={homepageMetaDescription} />
        <meta property="og:image" content={ogImage} />
        <meta
          property="og:image:alt"
          content="vLLM Semantic Router social preview"
        />
        <meta property="og:type" content="website" />
        <meta
          name="keywords"
          content="Mixture-of-Models runtime, preference-driven AI, open-source LLM router, multi-model routing, model orchestration, model selection, model cascade, Fusion API, micro-agent workflows, semantic router, policy-aware routing, vLLM"
        />
        <meta name="twitter:card" content="summary_large_image" />
        <meta name="twitter:title" content={homepageSocialTitle} />
        <meta name="twitter:description" content={homepageMetaDescription} />
        <meta name="twitter:image" content={ogImage} />
        <meta
          name="twitter:image:alt"
          content="vLLM Semantic Router social preview"
        />
        <script
          type="application/ld+json"
          dangerouslySetInnerHTML={{
            __html: JSON.stringify(homepageStructuredData),
          }}
        />
      </Head>
      <main className={styles.page} data-theme="dark">
        <SemanticTerrainHero />

        <div className={styles.bandGraphite}>
          <ValuePillars />
        </div>

        <div className={styles.bandBlack}>
          <IntegrationArchitecture />
        </div>

        <div className={styles.bandGraphite}>
          <ScrollReveal>
            <TestimonialsRail />
          </ScrollReveal>
        </div>

        <div className={styles.bandGraphite}>
          <CapabilitySection />
        </div>

        <div className={styles.bandGraphite}>
          <ScrollReveal delay={50}>
            <InstallQuickStartSection />
          </ScrollReveal>
        </div>

        <div className={styles.bandRaised}>
          <ScrollReveal delay={60}>
            <PaperFigureShowcase />
          </ScrollReveal>
        </div>

        <div className={styles.bandBlack}>
          <MixtureOfModelsProofSection />
        </div>

        <div className={styles.bandBlack}>
          <UseCaseExplorer />
        </div>

        <div className={styles.bandGraphite}>
          <CompatibilityBand />
        </div>

        <div className={styles.bandBlack}>
          <ScrollReveal delay={40}>
            <ResearchPaperCarousel />
          </ScrollReveal>
        </div>

        <div className={styles.bandGraphite}>
          <ScrollReveal delay={40}>
            <TeamCarousel />
          </ScrollReveal>
        </div>

        <SponsorsSection />

        <div className={styles.bandGraphite}>
          <FinalCtaSection />
        </div>
      </main>
    </Layout>
  )
}
