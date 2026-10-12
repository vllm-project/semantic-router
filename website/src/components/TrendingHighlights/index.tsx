import React from 'react'
import Link from '@docusaurus/Link'
import Translate from '@docusaurus/Translate'
import useBaseUrl from '@docusaurus/useBaseUrl'
import { SectionLabel } from '@site/src/components/site/Chrome'
import styles from './index.module.css'

interface HighlightMetric {
  value: string
  label: string
}

interface Highlight {
  id: string
  dateIso: string
  dateLabel: string
  label: string
  title: string
  statement: string
  image?: string
  imageWidth?: number
  imageHeight?: number
  imageAlt?: string
  metrics?: HighlightMetric[]
  sourceLabel: string
  sourceUrl: string
  internal?: boolean
}

const highlights: Highlight[] = [
  {
    id: 'hf-model-downloads',
    dateIso: '2026-10-10',
    dateLabel: 'October 10, 2026',
    label: 'Hugging Face models',
    title: '143K+ downloads across open router models',
    statement:
      'As of October 10, 2026, vLLM Semantic Router models on Hugging Face total about 143,000 downloads across 96 models in 6 collections—Decision, Vela, and MoM families.',
    metrics: [
      { value: '143K+', label: 'Downloads' },
      { value: '96', label: 'Models' },
      { value: '6', label: 'Collections' },
    ],
    sourceLabel: 'Hugging Face · vllm-sr collections',
    sourceUrl: 'https://huggingface.co/vllm-sr/collections',
  },
  {
    id: 'decision-20-hf-trending',
    dateIso: '2026-10-06',
    dateLabel: 'October 6–7, 2026',
    label: '#1 trending collection',
    title: 'Decision 2.0 tops Hugging Face trending',
    statement:
      'Decision 2.0 reached #1 on Hugging Face trending collections, leading selected open decision models across every size range on the Jev Decision Index 0.3.',
    image: '/img/trending/decision-20-jev-index.jpg',
    imageWidth: 1024,
    imageHeight: 582,
    imageAlt:
      'Bar chart comparing Decision 2.0 models against other open decision models on the Jev Decision Index 0.3.',
    sourceLabel: 'Hugging Face collection · Decision 2.0',
    sourceUrl: 'https://huggingface.co/collections/vllm-sr/decision-20',
  },
  {
    id: 'decision-20-jev-author-upvote',
    dateIso: '2026-10-09',
    dateLabel: 'October 9, 2026',
    label: 'Community upvote',
    title: 'Jev Decision Index author upvotes Decision 2.0',
    statement:
      'multimodalart, author of the Jev Decision Index Space, upvoted the Decision 2.0 collection on Hugging Face—community recognition from the benchmark that tracks open decision models.',
    image: '/img/blog/vela-2-0/jev-index.png',
    imageWidth: 2448,
    imageHeight: 1496,
    imageAlt:
      'Jev Decision Index chart used with Decision models; the Index author upvoted the Decision 2.0 collection.',
    sourceLabel: 'Hugging Face collection · Decision 2.0',
    sourceUrl: 'https://huggingface.co/collections/vllm-sr/decision-20',
  },
  {
    id: 'vela-20-launch',
    dateIso: '2026-10-06',
    dateLabel: 'October 6, 2026',
    label: 'Vela 2.0 launch',
    title: 'Open foundation routing models',
    statement:
      'Vela 2.0 ships four open routing models that answer safety, domain, PII, and hallucination questions in one call—with span-level answers.',
    image: '/img/blog/vela2-hero.jpg',
    imageWidth: 1920,
    imageHeight: 1080,
    imageAlt: 'Vela 2.0: Open Foundation Routing Models.',
    sourceLabel: 'Launch post · Vela 2.0',
    sourceUrl: '/blog/vela-2-0-open-foundation-routing-models',
    internal: true,
  },
  {
    id: 'vela-20-alphasignal',
    dateIso: '2026-10-07',
    dateLabel: 'October 7, 2026',
    label: 'AlphaSignal coverage',
    title: 'Vela 2.0 covered by AlphaSignal',
    statement:
      'AlphaSignal reported on Vela 2.0: four open routing models that fold nine LLM router checks into a single call.',
    image: '/img/trending/vela-20-alphasignal.svg',
    imageWidth: 1200,
    imageHeight: 675,
    imageAlt: 'AlphaSignal and vLLM Semantic Router logos — Vela 2.0 press coverage.',
    sourceLabel: 'AlphaSignal · Vela 2.0',
    sourceUrl:
      'https://alphasignal.ai/news/vllm-s-vela-2-0-collapses-nine-ai-safety-checks-into-one-open-model',
  },
  {
    id: 'jev-decision-index',
    dateIso: '2026-09-17',
    dateLabel: 'September 17, 2026',
    label: 'Hugging Face Space',
    title: 'Jev Decision Index on Hugging Face',
    statement:
      'The community Jev Decision Index Space tracks open decision-model benchmarks and news, with Decision 2.0 models listed among them.',
    image: '/img/trending/jev-decision-index.png',
    imageWidth: 2400,
    imageHeight: 1260,
    imageAlt: 'Hugging Face Space card for the Jev Decision Index by multimodalart.',
    sourceLabel: 'Hugging Face Space · Jev Decision Index',
    sourceUrl: 'https://huggingface.co/spaces/multimodalart/jev-decision-index',
  },
  {
    id: 'hermes-v04',
    dateIso: '2026-09-24',
    dateLabel: 'September 24, 2026',
    label: 'v0.4 Hermes release',
    title: 'Decision and classification models in Hermes',
    statement:
      'v0.4 Hermes shipped open Decision models and Vela classification models inside one improving Mixture-of-Models system.',
    image: '/img/blog/vllm/2026-09-24-v0.4-hermes-release/hero.png',
    imageWidth: 1731,
    imageHeight: 909,
    imageAlt: 'vLLM Semantic Router v0.4 Hermes hero: many models, one improving system.',
    sourceLabel: 'Release post · v0.4 Hermes',
    sourceUrl: '/blog/v0.4-vllm-sr-hermes-release',
    internal: true,
  },
]

function revealFocusedCard(event: React.FocusEvent<HTMLAnchorElement>): void {
  const card = event.currentTarget.closest('article')
  const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches

  card?.scrollIntoView({
    behavior: prefersReducedMotion ? 'auto' : 'smooth',
    block: 'nearest',
    inline: 'center',
  })
}

interface HighlightSequenceProps {
  duplicate?: boolean
}

function HighlightCard({
  highlight,
  duplicate = false,
}: {
  highlight: Highlight
  duplicate?: boolean
}): JSX.Element {
  const imageUrl = useBaseUrl(highlight.image ?? '/img/hf-trending.svg')
  const sourceProps = {
    className: styles.sourceLink,
    tabIndex: duplicate ? -1 : undefined,
    onFocus: revealFocusedCard,
  }

  return (
    <article
      className={styles.card}
      aria-label={duplicate ? undefined : `${highlight.title}, ${highlight.dateLabel}`}
    >
      <div className={styles.cardHeader}>
        <span className={styles.cardLabel}>{highlight.label}</span>
        <span className={styles.name}>{highlight.title}</span>
        <time className={styles.date} dateTime={highlight.dateIso}>
          {highlight.dateLabel}
        </time>
      </div>

      {highlight.metrics
        ? (
            <div
              className={styles.metrics}
              aria-label="Hugging Face model metrics: downloads, models, collections, and likes"
            >
              {highlight.metrics.map(metric => (
                <div key={metric.label} className={styles.metric}>
                  <strong className={styles.metricValue}>{metric.value}</strong>
                  <span className={styles.metricLabel}>{metric.label}</span>
                </div>
              ))}
            </div>
          )
        : (
            <div className={styles.media}>
              <img
                src={imageUrl}
                alt=""
                width={highlight.imageWidth}
                height={highlight.imageHeight}
                loading="lazy"
              />
            </div>
          )}

      <p className={styles.statement}>{highlight.statement}</p>

      {highlight.internal
        ? (
            <Link {...sourceProps} to={highlight.sourceUrl}>
              <span>
                <Translate id="homepage.trending.source">Source</Translate>
              </span>
              <span className={styles.sourceName}>{highlight.sourceLabel}</span>
              <span className={styles.sourceArrow} aria-hidden="true">↗</span>
            </Link>
          )
        : (
            <a
              {...sourceProps}
              href={highlight.sourceUrl}
              target="_blank"
              rel="noopener noreferrer"
            >
              <span>
                <Translate id="homepage.trending.source">Source</Translate>
              </span>
              <span className={styles.sourceName}>{highlight.sourceLabel}</span>
              <span className={styles.sourceArrow} aria-hidden="true">↗</span>
            </a>
          )}
    </article>
  )
}

function HighlightSequence({ duplicate = false }: HighlightSequenceProps): JSX.Element {
  return (
    <div className={styles.sequence} aria-hidden={duplicate || undefined}>
      {highlights.map(highlight => (
        <HighlightCard key={highlight.id} highlight={highlight} duplicate={duplicate} />
      ))}
    </div>
  )
}

export default function TrendingHighlights(): JSX.Element {
  return (
    <section className={styles.section} aria-labelledby="trending-highlights-heading">
      <div className="site-shell-container">
        <header className={styles.header}>
          <SectionLabel>
            <Translate id="homepage.trending.label">Trending highlights</Translate>
          </SectionLabel>
          <h2 className={styles.title} id="trending-highlights-heading">
            <Translate id="homepage.trending.title">The momentum is already visible</Translate>
          </h2>
        </header>
      </div>

      <div className={styles.viewport}>
        <div className={styles.track}>
          <HighlightSequence />
          <HighlightSequence duplicate />
        </div>
      </div>
    </section>
  )
}
