import React, { useCallback, useRef } from 'react'
import Link from '@docusaurus/Link'
import Translate from '@docusaurus/Translate'
import PixelIcon from '@site/src/components/duo/PixelIcon'
import AsciiOcean, { quietRectsIn } from '@site/src/components/site/AsciiOcean'
import duo from '@site/src/components/duo/duo.module.css'
import styles from './index.module.css'

const FILM_DURATION = '2:55'

/* The masthead, after System Duo's: porcelain light, the promise as the one large thing, two actions, and behind them
 * an ASCII ocean whose schools and sharks swim around the copy. */
export default function SemanticTerrainHero(): JSX.Element {
  const masthead = useRef<HTMLElement>(null)
  const quietRects = useCallback(() => quietRectsIn(masthead.current), [])

  return (
    <section ref={masthead} className={styles.masthead}>
      <AsciiOcean className={styles.ocean} quietRects={quietRects} />
      <div className={styles.inner}>
        <h1 className={styles.title} data-ocean-quiet="text">
          <Translate
            id="homepage.hero.title"
            values={{
              beyond: (
                <em className={styles.accent}>
                  <Translate id="homepage.hero.beyond">beyond any one model.</Translate>
                </em>
              ),
            }}
          >
            {'Intelligence {beyond}'}
          </Translate>
        </h1>
        <p className={styles.slogan} data-ocean-quiet="text">
          <Translate
            id="homepage.hero.dek"
            values={{
              decisionLayer: (
                <em>
                  <Translate id="homepage.hero.decisionLayer">decision layer</Translate>
                </em>
              ),
            }}
          >
            {'An open, programmable {decisionLayer} for models and compute.'}
          </Translate>
        </p>
        <div className={styles.actions} data-hero-actions data-ocean-quiet>
          <a className={duo.btnPrimary} href="https://app.vllm-sr.ai/#try" target="_blank" rel="noreferrer">
            <PixelIcon name="bolt" size={20} className={duo.btnIcon} />
            <Translate id="homepage.hero.primaryCta">
              Try System Duo
            </Translate>
          </a>
          <Link className={duo.btnAlt} to="/docs/installation/">
            <PixelIcon name="code" size={20} className={duo.btnIcon} />
            <Translate id="homepage.hero.secondaryCta">
              Get started
            </Translate>
          </Link>
        </div>
        <p className={styles.quiet} data-ocean-quiet="text">
          <a href="#watch">
            <Translate id="homepage.hero.filmLink">Watch the film</Translate>
            <span aria-hidden="true"> · </span>
            <span className={styles.duration}>{FILM_DURATION}</span>
          </a>
        </p>
      </div>
    </section>
  )
}
