import React from 'react'
import Translate, { translate } from '@docusaurus/Translate'
import { SectionLabel } from '@site/src/components/site/Chrome'
import shared from './homepageShared.module.css'
import styles from './PerformanceBenchmarks.module.css'

type BenchmarkCard = {
  category: string
  multiplier: string
  headline: string
  primary: string
  secondary: string
  secondaryLabel: string
}

const benchmarkCards: BenchmarkCard[] = [
  {
    category: translate({
      id: 'homepage.benchmarks.signals.category',
      message: 'Signals',
    }),
    multiplier: '16',
    headline: translate({
      id: 'homepage.benchmarks.signals.headline',
      message: 'Signal families. One layer.',
    }),
    primary: '16',
    secondary: '1',
    secondaryLabel: translate({
      id: 'homepage.benchmarks.signals.secondary',
      message: 'monolithic router',
    }),
  },
  {
    category: translate({
      id: 'homepage.benchmarks.algorithms.category',
      message: 'Selection',
    }),
    multiplier: '12',
    headline: translate({
      id: 'homepage.benchmarks.algorithms.headline',
      message: 'Algorithms. Executable paths.',
    }),
    primary: '12',
    secondary: '1',
    secondaryLabel: translate({
      id: 'homepage.benchmarks.algorithms.secondary',
      message: 'static backend',
    }),
  },
  {
    category: translate({
      id: 'homepage.benchmarks.models.category',
      message: 'Model fleet',
    }),
    multiplier: 'N',
    headline: translate({
      id: 'homepage.benchmarks.models.headline',
      message: 'Heterogeneous LLMs. One API.',
    }),
    primary: translate({
      id: 'homepage.benchmarks.models.primary',
      message: 'MoM',
    }),
    secondary: translate({
      id: 'homepage.benchmarks.models.secondary',
      message: 'single model',
    }),
    secondaryLabel: '',
  },
  {
    category: translate({
      id: 'homepage.benchmarks.replay.category',
      message: 'Replay',
    }),
    multiplier: translate({
      id: 'homepage.benchmarks.replay.multiplier',
      message: 'Full',
    }),
    headline: translate({
      id: 'homepage.benchmarks.replay.headline',
      message: 'Audit every routing decision.',
    }),
    primary: translate({
      id: 'homepage.benchmarks.replay.primary',
      message: 'Summary + detail',
    }),
    secondary: translate({
      id: 'homepage.benchmarks.replay.secondary',
      message: 'blind proxy',
    }),
    secondaryLabel: '',
  },
]

const badgeItems = [
  translate({ id: 'homepage.benchmarks.badge.envoy', message: 'Envoy ExtProc' }),
  translate({ id: 'homepage.benchmarks.badge.go', message: 'Go router' }),
  translate({ id: 'homepage.benchmarks.badge.opensource', message: 'Apache 2.0' }),
  translate({ id: 'homepage.benchmarks.badge.vllm', message: 'vLLM ecosystem' }),
]

export default function PerformanceBenchmarks(): JSX.Element {
  return (
    <section className={shared.lightSection} aria-labelledby="performance-benchmarks-title">
      <div className={`site-shell-container ${shared.sectionInner}`}>
        <header className={shared.sectionHeader}>
          <span className={shared.sectionEyebrow}>
            <Translate id="homepage.benchmarks.eyebrow">Built for inference routing</Translate>
          </span>
          <h2 id="performance-benchmarks-title" className={shared.sectionTitle}>
            <Translate id="homepage.benchmarks.title">
              Purpose-built to route as fast as models evolve
            </Translate>
          </h2>
          <p className={shared.sectionSubtitle}>
            <Translate id="homepage.benchmarks.subtitle">
              Encoder signals, decision plugins, and Envoy ext_proc integration —
              without retrofitting a generic API gateway for LLM semantics.
            </Translate>
          </p>
        </header>

        <div className={styles.cardGrid}>
          {benchmarkCards.map(card => (
            <article key={card.category} className={styles.card}>
              <SectionLabel>{card.category}</SectionLabel>
              <div className={styles.multiplier}>{card.multiplier}</div>
              <h3>{card.headline}</h3>
              <div className={styles.compare}>
                <span className={styles.primary}>{card.primary}</span>
                <span className={styles.vs}>
                  <Translate id="homepage.benchmarks.vs">vs</Translate>
                </span>
                <span className={styles.secondary}>{card.secondary}</span>
              </div>
              {card.secondaryLabel && (
                <span className={styles.secondaryLabel}>{card.secondaryLabel}</span>
              )}
            </article>
          ))}
        </div>

        <div className={styles.badgeRow}>
          {badgeItems.map(item => (
            <span key={item} className={styles.badge}>
              {item}
            </span>
          ))}
        </div>
      </div>
    </section>
  )
}
