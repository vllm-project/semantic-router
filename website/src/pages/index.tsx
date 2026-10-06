import React from 'react'
import clsx from 'clsx'
import Head from '@docusaurus/Head'
import Link from '@docusaurus/Link'
import Layout from '@theme/Layout'
import Translate, { translate } from '@docusaurus/Translate'
import useDocusaurusContext from '@docusaurus/useDocusaurusContext'
import useBaseUrl from '@docusaurus/useBaseUrl'
import IntegrationArchitecture from '@site/src/components/homepage/IntegrationArchitecture'
import EcosystemSection from '@site/src/components/homepage/EcosystemSection'
import AcknowledgementsSection from '@site/src/components/AcknowledgementsSection'
import InstallQuickStartSection from '@site/src/components/InstallQuickStartSection'
import YouTubeSection from '@site/src/components/YouTubeSection'
import ResearchPaperCarousel from '@site/src/components/ResearchPaperCarousel'
import TeamCarousel from '@site/src/components/TeamCarousel'
import TestimonialsRail from '@site/src/components/TestimonialsRail'
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
  message: 'Intelligence Beyond Any One Model',
})
const homepageMetaDescription = translate({
  id: 'homepage.meta.description',
  message:
    'An open, programmable decision layer for models and compute.',
})
const homepageSocialTitle = translate({
  id: 'homepage.meta.socialTitle',
  message: 'Intelligence Beyond Any One Model | vLLM Semantic Router',
})

const heroStats = [
  {
    label: translate({
      id: 'homepage.stats.signals.label',
      message: 'Signals',
    }),
    value: '20',
    description: translate({
      id: 'homepage.stats.signals.description',
      message:
        'Context, intent, safety, preferences, and system state.',
    }),
  },
  {
    label: translate({
      id: 'homepage.stats.algorithms.label',
      message: 'Algorithms',
    }),
    value: '16',
    description: translate({
      id: 'homepage.stats.algorithms.description',
      message:
        '11 selection algorithms and 5 loopers.',
    }),
  },
  {
    label: translate({ id: 'homepage.stats.papers.label', message: 'Papers' }),
    value: String(paperCount).padStart(2, '0'),
    description: translate(
      {
        id: 'homepage.stats.papers.description',
        message:
          '{count} papers on routing, safety, and inference.',
      },
      { count: paperCount },
    ),
  },
]

/* The four beats of the routing decision. The gate stage is the claim: hard
 * constraints remove paths, and only what survives is ranked. */
const sovereigntyStages = [
  {
    id: 'request',
    label: translate({ id: 'homepage.sovereignty.stage.request', message: 'Request' }),
    detail: translate({ id: 'homepage.sovereignty.stage.request.detail', message: 'Identity + context' }),
    gate: false,
  },
  {
    id: 'constraints',
    label: translate({ id: 'homepage.sovereignty.stage.constraints', message: 'Hard constraints' }),
    detail: translate({ id: 'homepage.sovereignty.stage.constraints.detail', message: 'Residency · locality · auth' }),
    gate: true,
  },
  {
    id: 'eligible',
    label: translate({ id: 'homepage.sovereignty.stage.eligible', message: 'Eligible pool' }),
    detail: translate({ id: 'homepage.sovereignty.stage.eligible.detail', message: 'Approved paths only' }),
    gate: false,
  },
  {
    id: 'rank',
    label: translate({ id: 'homepage.sovereignty.stage.rank', message: 'Rank' }),
    detail: translate({ id: 'homepage.sovereignty.stage.rank.detail', message: 'Quality · latency · cost' }),
    gate: false,
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
      message: 'Combine their strengths.',
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
      message: 'GPUs differ in speed and capacity.',
    }),
    unified: translate({
      id: 'homepage.capabilities.compute.value',
      message: 'Choose among your configured backends.',
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
      message: 'Edge, private, and cloud.',
    }),
    unified: translate({
      id: 'homepage.capabilities.location.value',
      message: 'Stay within approved locations.',
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
      message: 'Priorities change by task.',
    }),
    unified: translate({
      id: 'homepage.capabilities.preference.value',
      message: 'Set quality, latency, and cost priorities.',
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

type ExampleProduct = {
  label: string
  to?: string
}

function ExampleProducts({ products }: { products: ExampleProduct[] }): JSX.Element {
  return (
    <span className="site-prose">
      {products.map((product, index) => (
        <React.Fragment key={product.label}>
          {index > 0 && ', '}
          {product.to ? <Link to={product.to}>{product.label}</Link> : product.label}
        </React.Fragment>
      ))}
    </span>
  )
}

const alternativeComparison = [
  {
    capability: translate({
      id: 'homepage.alternatives.examples.capability',
      message: 'Examples',
    }),
    semanticRouter: 'vLLM Semantic Router',
    aiGateway: (
      <ExampleProducts
        products={[
          { label: 'Agent Router', to: '/docs/installation/k8s/ai-gateway' },
          { label: 'LiteLLM' },
          { label: 'agentgateway', to: '/docs/installation/k8s/agentgateway' },
        ]}
      />
    ),
    llmd: (
      <ExampleProducts
        products={[
          { label: 'llm-d', to: '/docs/installation/k8s/llm-d' },
          { label: 'vLLM Router' },
          { label: 'AIBrix gateway', to: '/docs/installation/k8s/aibrix' },
        ]}
      />
    ),
  },
  {
    capability: translate({
      id: 'homepage.alternatives.decides.capability',
      message: 'What it decides',
    }),
    semanticRouter: translate({
      id: 'homepage.alternatives.decides.semanticRouter',
      message: 'Models and policy for each request',
    }),
    aiGateway: translate({
      id: 'homepage.alternatives.decides.aiGateway',
      message: 'How a request reaches a backend',
    }),
    llmd: translate({
      id: 'homepage.alternatives.decides.llmd',
      message: 'A healthy replica in the chosen pool',
    }),
  },
  {
    capability: translate({
      id: 'homepage.alternatives.reads.capability',
      message: 'What it reads',
    }),
    semanticRouter: translate({
      id: 'homepage.alternatives.reads.semanticRouter',
      message: 'Request meaning and policy',
    }),
    aiGateway: translate({
      id: 'homepage.alternatives.reads.aiGateway',
      message: 'Protocol, credentials, rate limits',
    }),
    llmd: translate({
      id: 'homepage.alternatives.reads.llmd',
      message: 'Load, prefix-cache locality, replica health',
    }),
  },
  {
    capability: translate({
      id: 'homepage.alternatives.owns.capability',
      message: 'What it owns',
    }),
    semanticRouter: translate({
      id: 'homepage.alternatives.owns.semanticRouter',
      message: 'Model selection and collaboration',
    }),
    aiGateway: translate({
      id: 'homepage.alternatives.owns.aiGateway',
      message: 'Provider access and traffic control',
    }),
    llmd: translate({
      id: 'homepage.alternatives.owns.llmd',
      message: 'Endpoint selection inside a pool',
    }),
  },
  {
    capability: translate({
      id: 'homepage.alternatives.runs.capability',
      message: 'Where it runs',
    }),
    semanticRouter: translate({
      id: 'homepage.alternatives.runs.semanticRouter',
      message: 'An Envoy ExtProc filter',
    }),
    aiGateway: translate({
      id: 'homepage.alternatives.runs.aiGateway',
      message: 'The data plane',
    }),
    llmd: translate({
      id: 'homepage.alternatives.runs.llmd',
      message: 'A pool scheduler, such as llm-d',
    }),
  },
  {
    capability: translate({
      id: 'homepage.alternatives.receipt.capability',
      message: 'Decision receipt',
    }),
    semanticRouter: translate({
      id: 'homepage.alternatives.receipt.semanticRouter',
      message: 'x-vsr-selected-model',
    }),
    aiGateway: translate({
      id: 'homepage.alternatives.receipt.aiGateway',
      message: 'Varies by implementation',
    }),
    llmd: translate({
      id: 'homepage.alternatives.receipt.llmd',
      message: 'Varies by implementation',
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
          <header className={`site-section-intro ${styles.capabilityHeading}`}>
            <SectionLabel>
              <Translate id="homepage.capabilities.label">Architecture</Translate>
            </SectionLabel>
            <h2 id="mixture-architecture-title">
              <Translate id="homepage.capabilities.heading">
                Your models. Your rules.
              </Translate>
            </h2>
            <p>
              <Translate id="homepage.capabilities.description">
                Choose models and compute for each request.
              </Translate>
            </p>
          </header>

          <div className={styles.capabilityFrame}>
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

function AlternativesSection(): JSX.Element {
  return (
    <section
      className={styles.capabilitySection}
      aria-labelledby="alternatives-title"
    >
      <div className="site-shell-container">
        <ScrollReveal>
          <header className={`site-section-intro ${styles.capabilityHeading}`}>
            <SectionLabel>
              <Translate id="homepage.alternatives.label">
                Where it fits
              </Translate>
            </SectionLabel>
            <h2 id="alternatives-title">
              <Translate id="homepage.alternatives.heading">
                Fits your stack.
              </Translate>
            </h2>
            <p>
              <Translate id="homepage.alternatives.description">
                Your harness runs the agent loop and tools. The Router chooses
                models and backends.
              </Translate>
            </p>
          </header>

          <div className={styles.capabilityFrame}>
            <div
              className={`${styles.architectureMatrix} ${styles.alternativesMatrix}`}
              role="table"
              aria-label={translate({
                id: 'homepage.alternatives.table.aria',
                message:
                  'Semantic Router compared with an AI Gateway and an Inference Router',
              })}
            >
              <div className={styles.matrixHeader} role="row">
                <span role="columnheader">
                  <Translate id="homepage.alternatives.table.capability">
                    Capability
                  </Translate>
                </span>
                <span role="columnheader">
                  <Translate id="homepage.alternatives.table.semanticRouter">
                    Semantic Router
                  </Translate>
                </span>
                <span role="columnheader">
                  <Translate id="homepage.alternatives.table.aiGateway">
                    AI Gateway
                  </Translate>
                </span>
                <span role="columnheader">
                  <Translate id="homepage.alternatives.table.llmd">
                    Inference Router
                  </Translate>
                </span>
              </div>

              {alternativeComparison.map(item => (
                <div key={item.capability} className={styles.matrixRow} role="row">
                  <div className={styles.matrixDimension} role="rowheader">
                    <strong>{item.capability}</strong>
                  </div>
                  <div className={styles.matrixUnified} role="cell">
                    <span className={styles.matrixMobileLabel}>
                      <Translate id="homepage.alternatives.table.semanticRouter">
                        Semantic Router
                      </Translate>
                    </span>
                    <p>{item.semanticRouter}</p>
                  </div>
                  <div className={styles.matrixFragmented} role="cell">
                    <span className={styles.matrixMobileLabel}>
                      <Translate id="homepage.alternatives.table.aiGateway">
                        AI Gateway
                      </Translate>
                    </span>
                    <p>{item.aiGateway}</p>
                  </div>
                  <div className={styles.matrixFragmented} role="cell">
                    <span className={styles.matrixMobileLabel}>
                      <Translate id="homepage.alternatives.table.llmd">
                        Inference Router
                      </Translate>
                    </span>
                    <p>{item.llmd}</p>
                  </div>
                </div>
              ))}
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
                Mixture-of-Models
              </Translate>
            </SectionLabel>
            <div>
              <h2 id="mom-proof-title">
                <Translate id="homepage.momProof.title">
                  Stronger models, together.
                </Translate>
              </h2>
              <p>
                <Translate id="homepage.momProof.description">
                  One API brings open and closed models together.
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
                    One call. More than one model.
                  </Translate>
                </h3>
                <p>
                  <Translate id="homepage.momProof.architectureCopy">
                    Bounded model collaboration. Your harness keeps the agent loop.
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

/* Residency and authorization are eliminated as hard constraints before any
 * ranking happens. Deliberately no claim that data never leaves your estate —
 * docs/overview/use-cases.md is explicit that "local" is not an end-to-end
 * privacy guarantee, and the homepage should not say otherwise. */
function DataSovereigntySection(): JSX.Element {
  return (
    <section className={styles.sovereigntySection}>
      <div className="site-shell-container">
        <ScrollReveal>
          <div className={styles.sovereigntyFrame}>
            <div className={`site-section-intro ${styles.sovereigntyCopy}`}>
              <SectionLabel>
                <Translate id="homepage.sovereignty.label">Data sovereignty</Translate>
              </SectionLabel>
              <h2>
                <Translate id="homepage.sovereignty.title">
                  Your data. Your boundaries.
                </Translate>
              </h2>
              <p>
                <Translate id="homepage.sovereignty.description">
                  Enforce residency, locality, and authorization before choosing a model.
                </Translate>
              </p>
            </div>

            <ol className={styles.constraintFlow}>
              {sovereigntyStages.map((stage, index) => (
                <li
                  key={stage.id}
                  className={clsx(styles.constraintStage, {
                    [styles.constraintStageGate]: stage.gate,
                  })}
                  style={{ '--stage-delay': `${index * 0.5}s` } as React.CSSProperties}
                >
                  <span className={styles.constraintStageIndex}>
                    {String(index + 1).padStart(2, '0')}
                  </span>
                  <strong>{stage.label}</strong>
                  <span className={styles.constraintStageDetail}>{stage.detail}</span>
                </li>
              ))}
            </ol>

            <div className={styles.constraintFooter}>
              <p className={styles.constraintNote}>
                <Translate id="homepage.sovereignty.note">
                  Requests fail closed when no approved route remains.
                </Translate>
              </p>
              <PillLink className={styles.sovereigntyCta} to="/docs/overview/signal-driven-decisions" muted>
                <Translate id="homepage.sovereignty.cta">Explore routing policies</Translate>
              </PillLink>
            </div>
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
                  Build beyond one model.
                </Translate>
              </h2>
              <p>
                <Translate id="homepage.finalCta.description">
                  Connect your harness. Choose your models. Set your rules.
                </Translate>
              </p>
            </div>
            <div className={styles.finalCtaActions}>
              <PillLink
                className={styles.finalCtaPrimary}
                href="https://app.vllm-sr.ai/playground"
                rel="noreferrer"
                target="_blank"
              >
                <Translate id="homepage.finalCta.playground">Try the Playground</Translate>
              </PillLink>
              <PillLink to="/docs/overview/use-cases" muted>
                <Translate id="homepage.finalCta.docs">Explore use cases</Translate>
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
  const filmPoster = useBaseUrl('/videos/vllm-sr-intro/vllm-sr-intro-poster.webp')
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
      'https://huggingface.co/vllm-sr',
    ],
  }

  return (
    <Layout title={homepageMetaTitle} description={homepageMetaDescription}>
      <Head>
        <link rel="preload" as="image" href={filmPoster} fetchPriority="high" />
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
          content="programmable decision layer, agent harness, models and compute, Mixture-of-Models, open-source LLM router, multi-model routing, model selection, bounded model collaboration, semantic router, policy-aware routing, vLLM"
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
      <main className={styles.page}>
        <SemanticTerrainHero />

        <div className={styles.bandGraphite}>
          <ScrollReveal>
            <TestimonialsRail />
          </ScrollReveal>
        </div>

        <div className={styles.bandGraphite}>
          <ScrollReveal delay={50}>
            <InstallQuickStartSection />
          </ScrollReveal>
        </div>

        <div className={styles.bandRaised}>
          <ScrollReveal delay={50}>
            <YouTubeSection />
          </ScrollReveal>
        </div>

        <div className={styles.bandBlack}>
          <CapabilitySection />
        </div>

        <div className={styles.bandGraphite}>
          <IntegrationArchitecture />
        </div>

        <div className={styles.bandBlack}>
          <AlternativesSection />
        </div>

        <div className={styles.bandBlack}>
          <MixtureOfModelsProofSection />
        </div>

        <div className={styles.bandBlack}>
          <EcosystemSection />
        </div>

        <div className={styles.bandRaised}>
          <DataSovereigntySection />
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

        <div className={styles.bandBlack}>
          <ScrollReveal delay={40}>
            <AcknowledgementsSection />
          </ScrollReveal>
        </div>

        <div className={styles.bandGraphite}>
          <FinalCtaSection />
        </div>
      </main>
    </Layout>
  )
}
