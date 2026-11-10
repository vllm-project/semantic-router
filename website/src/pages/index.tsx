import React, { useEffect } from 'react'
import Head from '@docusaurus/Head'
import Link from '@docusaurus/Link'
import { HtmlClassNameProvider } from '@docusaurus/theme-common'
import Layout from '@theme/Layout'
import Translate, { translate } from '@docusaurus/Translate'
import useDocusaurusContext from '@docusaurus/useDocusaurusContext'
import Architecture from '@site/src/components/homepage/Architecture'
import RuntimeModes from '@site/src/components/homepage/RuntimeModes'
import SovereigntyAI from '@site/src/components/homepage/SovereigntyAI'
import EcosystemSection from '@site/src/components/homepage/EcosystemSection'
import InstallQuickStartSection from '@site/src/components/InstallQuickStartSection'
import YouTubeSection from '@site/src/components/YouTubeSection'
import ResearchPaperCarousel from '@site/src/components/ResearchPaperCarousel'
import TeamCarousel from '@site/src/components/TeamCarousel'
import TestimonialsRail from '@site/src/components/TestimonialsRail'
import TrendingHighlights from '@site/src/components/TrendingHighlights'
import { SITE_SOCIAL_PREVIEW_IMAGE_PATH } from '@site/src/data/socialPreview'
import SemanticTerrainHero from '@site/src/components/site/SemanticTerrainHero'
import ScrollReveal from '@site/src/components/site/ScrollReveal'
import { SectionLabel } from '@site/src/components/site/Chrome'
import Ground from '@site/src/components/duo/Ground'
import PixelIcon from '@site/src/components/duo/PixelIcon'
import Stars from '@site/src/components/duo/Stars'
import { DawnInterlude, EmberInterlude, NightInterlude } from '@site/src/components/duo/Interlude'
import styles from './index.module.css'

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

/* The close, as System Duo's mission: a dusk with pixel stars falls into ink, and the footer continues it. */
function FinalCtaSection(): JSX.Element {
  return (
    <section id="get-started" className={styles.mission} aria-labelledby="get-started-title">
      <div className={styles.dusk} aria-hidden="true">
        <Stars className={styles.stars} />
      </div>
      <div className={styles.missionIn}>
        <ScrollReveal>
          <div className={styles.missionGrid}>
            <div>
              <SectionLabel className={styles.missionLabel}>
                <Translate id="homepage.finalCta.label">Make it yours</Translate>
              </SectionLabel>
              <h2 id="get-started-title" className={styles.statement}>
                <Translate
                  id="homepage.finalCta.title"
                  values={{ count: <em><Translate id="homepage.finalCta.count">count</Translate></em> }}
                >
                  {'Make every request {count}.'}
                </Translate>
              </h2>
              <p className={styles.why}>
                <Translate id="homepage.finalCta.description">
                  Explore intelligent routing in System Duo. Bring it to your own stack when you're ready.
                </Translate>
              </p>
            </div>
            <ul className={styles.principles}>
              <li>
                <PixelIcon name="bolt" size={32} className={styles.principleIcon} />
                <a href="https://app.vllm-sr.ai/#try" target="_blank" rel="noreferrer">
                  <h3><Translate id="homepage.finalCta.playground">Try System Duo</Translate></h3>
                  <p><Translate id="homepage.finalCta.playgroundDetail">See how requests find the right model.</Translate></p>
                </a>
              </li>
              <li>
                <PixelIcon name="code" size={32} className={styles.principleIcon} />
                <Link to="/docs/installation/">
                  <h3><Translate id="homepage.finalCta.docs">Run it yourself</Translate></h3>
                  <p><Translate id="homepage.finalCta.docsDetail">From installation to your first request.</Translate></p>
                </Link>
              </li>
            </ul>
          </div>
        </ScrollReveal>
      </div>
    </section>
  )
}

// The navbar turns from clear to frosted paper after a little scroll; its pill appears once the hero's actions pass.
function useHomeChrome(): void {
  useEffect(() => {
    const root = document.documentElement
    const actions = document.querySelector('[data-field-calm]')
    let frame = 0
    const update = () => {
      frame = 0
      root.toggleAttribute('data-duo-scrolled', window.scrollY > 8)
      root.toggleAttribute('data-duo-cta', !actions || actions.getBoundingClientRect().bottom < 80)
    }
    const onScroll = () => {
      if (!frame) frame = window.requestAnimationFrame(update)
    }
    update()
    window.addEventListener('scroll', onScroll, { passive: true })
    return () => {
      window.removeEventListener('scroll', onScroll)
      if (frame) window.cancelAnimationFrame(frame)
      root.removeAttribute('data-duo-scrolled')
      root.removeAttribute('data-duo-cta')
    }
  }, [])
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
      'https://huggingface.co/vllm-sr',
    ],
  }
  useHomeChrome()

  return (
    <HtmlClassNameProvider className="duo-home">
      <Layout title={homepageMetaTitle} description={homepageMetaDescription}>
        <Head>
          {/* In the server-rendered head, before the first paint: reveals hide sections only when JavaScript runs. */}
          <script>{'document.documentElement.setAttribute(\'data-js\', \'\')'}</script>
          {/* The same root URLs that the site-wide cormorant.css resolves, in every locale. */}
          <link rel="preload" as="font" type="font/woff2" href="/fonts/CormorantGaramond.woff2" crossOrigin="anonymous" />
          <link rel="preload" as="font" type="font/woff2" href="/fonts/CormorantGaramond-Italic.woff2" crossOrigin="anonymous" />
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

          <Ground variant="blueprint">
            <ScrollReveal>
              <InstallQuickStartSection />
            </ScrollReveal>
          </Ground>

          <DawnInterlude />

          <Ground variant="band" tone="blue" rule seed={2}>
            <Architecture />
          </Ground>

          <Ground variant="blueprint">
            <RuntimeModes />
          </Ground>

          <NightInterlude />

          <Ground variant="paper" rule seed={3}>
            <ScrollReveal>
              <YouTubeSection />
            </ScrollReveal>
          </Ground>

          <Ground variant="band" tone="ember" rule seed={4}>
            <SovereigntyAI />
          </Ground>

          <Ground variant="blueprint">
            <ScrollReveal>
              <ResearchPaperCarousel />
            </ScrollReveal>
          </Ground>

          <EmberInterlude />

          <Ground variant="paper" rule seed={5}>
            <ScrollReveal>
              <TestimonialsRail />
            </ScrollReveal>
          </Ground>

          <Ground variant="blueprint">
            <ScrollReveal>
              <TeamCarousel />
            </ScrollReveal>
          </Ground>

          <Ground variant="paper">
            <ScrollReveal>
              <TrendingHighlights />
            </ScrollReveal>
          </Ground>

          <Ground variant="band" tone="violet" rule seed={6}>
            <ScrollReveal>
              <EcosystemSection />
            </ScrollReveal>
          </Ground>

          <FinalCtaSection />
        </main>
      </Layout>
    </HtmlClassNameProvider>
  )
}
