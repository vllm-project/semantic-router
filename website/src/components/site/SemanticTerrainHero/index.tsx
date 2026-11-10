import React, { useEffect, useRef } from 'react'
import Link from '@docusaurus/Link'
import Translate from '@docusaurus/Translate'
import PixelIcon from '@site/src/components/duo/PixelIcon'
import { mountField } from '@site/src/components/duo/field'
import duo from '@site/src/components/duo/duo.module.css'
import styles from './index.module.css'

const FILM_DURATION = '2:55'

function BlockField(): JSX.Element {
  const ref = useRef<HTMLDivElement>(null)
  useEffect(() => (ref.current ? mountField(ref.current) : undefined), [])

  return (
    <div ref={ref} className={styles.field} aria-hidden="true">
      <canvas />
    </div>
  )
}

/* The masthead, after System Duo's: porcelain light, the promise as the one large thing, two actions, and under them
 * a field of blocks that denoises into one horizon. */
export default function SemanticTerrainHero(): JSX.Element {
  return (
    <section className={styles.masthead} data-field-host>
      <BlockField />
      <div className={styles.inner}>
        <h1 className={styles.title}>
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
        <p className={styles.slogan}>
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
        <div className={styles.actions} data-field-calm>
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
        <p className={styles.quiet}>
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
