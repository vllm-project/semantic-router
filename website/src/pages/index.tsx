import React from 'react'
import Head from '@docusaurus/Head'
import Link from '@docusaurus/Link'
import { FiArrowUpRight, FiGlobe, FiTerminal } from 'react-icons/fi'
import Layout from '@theme/Layout'
import Translate, { translate } from '@docusaurus/Translate'
import useDocusaurusContext from '@docusaurus/useDocusaurusContext'
import useBaseUrl from '@docusaurus/useBaseUrl'
import ArchitectureCapabilities from '@site/src/components/homepage/ArchitectureCapabilities'
import RuntimeModes from '@site/src/components/homepage/RuntimeModes'
import SovereigntyAI from '@site/src/components/homepage/SovereigntyAI'
import EcosystemSection from '@site/src/components/homepage/EcosystemSection'
import AcknowledgementsSection from '@site/src/components/AcknowledgementsSection'
import InstallQuickStartSection from '@site/src/components/InstallQuickStartSection'
import YouTubeSection from '@site/src/components/YouTubeSection'
import ResearchPaperCarousel from '@site/src/components/ResearchPaperCarousel'
import TeamCarousel from '@site/src/components/TeamCarousel'
import TestimonialsRail from '@site/src/components/TestimonialsRail'
import TrendingHighlights from '@site/src/components/TrendingHighlights'
import { researchPapers } from '@site/src/data/researchContent'
import { SITE_SOCIAL_PREVIEW_IMAGE_PATH } from '@site/src/data/socialPreview'
import SemanticTerrainHero from '@site/src/components/site/SemanticTerrainHero'
import ScrollReveal from '@site/src/components/site/ScrollReveal'
import { SectionLabel } from '@site/src/components/site/Chrome'
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

function FinalCtaSection(): JSX.Element {
  return (
    <section id="get-started" className={styles.finalCtaSection} aria-labelledby="get-started-title">
      <div className="site-shell-container">
        <ScrollReveal>
          <div className={styles.finalCtaFrame}>
            <div className={styles.finalCtaCopy}>
              <SectionLabel>
                <Translate id="homepage.finalCta.label">Make it yours</Translate>
              </SectionLabel>
              <h2 id="get-started-title">
                <Translate id="homepage.finalCta.title">Make every request count.</Translate>
              </h2>
              <p>
                <Translate id="homepage.finalCta.description">
                  Explore intelligent routing in the Playground. Bring it to your own stack when you're ready.
                </Translate>
              </p>
            </div>
            <div className={styles.finalCtaActions}>
              <Link
                className={styles.finalCtaLink}
                href="https://app.vllm-sr.ai/playground"
                rel="noreferrer"
                target="_blank"
              >
                <span className={styles.finalCtaIcon} aria-hidden="true"><FiGlobe /></span>
                <span className={styles.finalCtaLinkCopy}>
                  <strong><Translate id="homepage.finalCta.playground">Try the Playground</Translate></strong>
                  <span><Translate id="homepage.finalCta.playgroundDetail">See how requests find the right model.</Translate></span>
                </span>
                <FiArrowUpRight className={styles.finalCtaArrow} aria-hidden="true" />
              </Link>
              <Link className={`${styles.finalCtaLink} ${styles.finalCtaInstall}`} to="/docs/installation/">
                <span className={styles.finalCtaIcon} aria-hidden="true"><FiTerminal /></span>
                <span className={styles.finalCtaLinkCopy}>
                  <strong><Translate id="homepage.finalCta.docs">Run it yourself</Translate></strong>
                  <span><Translate id="homepage.finalCta.docsDetail">From installation to your first request.</Translate></span>
                </span>
                <FiArrowUpRight className={styles.finalCtaArrow} aria-hidden="true" />
              </Link>
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
          <ScrollReveal delay={40}>
            <TrendingHighlights />
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

        <div className={styles.bandArchitecture}>
          <ArchitectureCapabilities stats={heroStats} />
        </div>

        <div className={styles.bandRuntime}>
          <RuntimeModes />
        </div>

        <div className={styles.bandRaised}>
          <SovereigntyAI />
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
            <EcosystemSection />
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
