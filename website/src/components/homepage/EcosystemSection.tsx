import React from 'react'
import Link from '@docusaurus/Link'
import Translate from '@docusaurus/Translate'
import useBaseUrl from '@docusaurus/useBaseUrl'
import ThemedImage from '@theme/ThemedImage'
import styles from './EcosystemSection.module.css'

/**
 * Homepage teaser for Ecosystem & partnerships. The full logo grid lives
 * on the Community page (/community/ecosystem) as the single source of
 * truth; this band keeps the wordmark, title, and tagline and deep-links
 * there. The section keeps id="ecosystem" so existing /#ecosystem links
 * (for example from the repository README) keep working.
 */
export default function EcosystemSection(): React.JSX.Element {
  const assetPath = useBaseUrl('/img/ecosystem/')

  return (
    <section id="ecosystem" className={styles.section} aria-labelledby="ecosystem-title">
      <div
        className={styles.atmosphere}
        style={{ backgroundImage: `url(${assetPath}atmosphere.webp)` }}
        aria-hidden="true"
      />
      <div className={styles.content}>
        <header className={styles.heading}>
          <ThemedImage
            className={styles.wordmark}
            sources={{
              light: `${assetPath}vllm-sr-wordmark-light.png`,
              dark: `${assetPath}vllm-sr-wordmark-dark.png`,
            }}
            alt="vLLM Semantic Router"
            width={2160}
            height={690}
            loading="lazy"
          />
          <h2 id="ecosystem-title">
            <Translate id="homepage.ecosystem.title">Ecosystem & partnerships</Translate>
          </h2>
          <p>
            <Translate
              id="homepage.ecosystem.description"
              values={{
                decisionLayer: (
                  <strong>
                    <Translate id="homepage.ecosystem.decisionLayer">decision layer</Translate>
                  </strong>
                ),
              }}
            >
              {'An open, programmable {decisionLayer} for models and compute.'}
            </Translate>
          </p>
        </header>

        <div className={styles.footer}>
          <p>
            <Translate id="homepage.ecosystem.growing">
              And an open community still growing.
            </Translate>
          </p>
          <Link to="/community/ecosystem">
            <Translate id="homepage.ecosystem.explore">Explore the ecosystem</Translate>
            <span aria-hidden="true">↗</span>
          </Link>
        </div>
      </div>
    </section>
  )
}
