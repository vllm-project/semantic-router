import React from 'react'
import Translate, { translate } from '@docusaurus/Translate'
import ScrollReveal from '@site/src/components/site/ScrollReveal'
import shared from './homepageShared.module.css'
import styles from './ValuePillars.module.css'

const pillars = [
  {
    title: translate({
      id: 'homepage.pillars.signal.title',
      message: 'Signal-driven',
    }),
    stat: '20',
    statLabel: translate({
      id: 'homepage.pillars.signal.stat',
      message: 'signal families',
    }),
    description: translate({
      id: 'homepage.pillars.signal.description',
      message:
        'Combine request, safety, intent, preference, and system signals inside each recipe.',
    }),
  },
  {
    title: translate({
      id: 'homepage.pillars.dropin.title',
      message: 'Drop-in',
    }),
    stat: '1',
    statLabel: translate({
      id: 'homepage.pillars.dropin.stat',
      message: 'OpenAI-compatible API',
    }),
    description: translate({
      id: 'homepage.pillars.dropin.description',
      message:
        'Send OpenAI-compatible requests through Envoy ExtProc or the local vllm-sr runtime.',
    }),
  },
  {
    title: translate({
      id: 'homepage.pillars.observable.title',
      message: 'Observable',
    }),
    stat: '16',
    statLabel: translate({
      id: 'homepage.pillars.observable.stat',
      message: 'selectors and loopers',
    }),
    description: translate({
      id: 'homepage.pillars.observable.description',
      message:
        'Choose from 11 selection algorithms and 5 loopers, then inspect routing metadata and metrics when enabled.',
    }),
  },
]

export default function ValuePillars(): JSX.Element {
  return (
    <section className={shared.bandSection} aria-labelledby="value-pillars-title">
      <div className={`site-shell-container ${shared.sectionInner}`}>
        <ScrollReveal>
          <header className={`${shared.sectionHeader} ${shared.sectionHeaderWide}`}>
            <span className={shared.eyebrow}>
              <Translate id="homepage.pillars.eyebrow">Why Semantic Router</Translate>
            </span>
            <h2 id="value-pillars-title" className={shared.sectionTitle}>
              <Translate id="homepage.pillars.title">Intelligent multi-model routing</Translate>
            </h2>
            <p className={shared.sectionSubtitle}>
              <Translate id="homepage.pillars.subtitle">Deploy fast, route by signal, and enable the observability your operation needs.</Translate>
            </p>
          </header>
        </ScrollReveal>

        <div className={styles.grid}>
          {pillars.map((pillar, index) => (
            <ScrollReveal key={pillar.title} delay={index * 90}>
              <article className={styles.card}>
                <div className={styles.cardTop}>
                  <h3>{pillar.title}</h3>
                  <div className={styles.statLockup}>
                    <span className={styles.statValue}>{pillar.stat}</span>
                    <span className={styles.statLabel}>{pillar.statLabel}</span>
                  </div>
                </div>
                <p>{pillar.description}</p>
              </article>
            </ScrollReveal>
          ))}
        </div>
      </div>
    </section>
  )
}
