import React from 'react'
import Translate from '@docusaurus/Translate'
import { SectionLabel } from '@site/src/components/site/Chrome'
import styles from './index.module.css'

interface Testimonial {
  image: string
  kind: 'quote' | 'perspective'
  label: string
  name: string
  role: string
  sourceLabel: string
  sourceUrl: string
  statement: string
}

const testimonials: Testimonial[] = [
  {
    image: '/img/testimonials/satya-nadella.jpg',
    kind: 'quote',
    label: 'Meta model',
    name: 'Satya Nadella',
    role: 'Microsoft Chairman & CEO',
    sourceLabel: 'Morgan Stanley TMT Conference · 2026',
    sourceUrl: 'https://www.microsoft.com/en-us/investor/events/fy-2026/morgan-stanley-tmt-conference',
    statement: 'Think of it as a meta model, right? That’s the most important thing.',
  },
  {
    image: '/img/testimonials/clement-delangue.png',
    kind: 'quote',
    label: 'Open-source routers',
    name: 'Clément Delangue',
    role: 'Hugging Face Co-founder & CEO',
    sourceLabel: 'Public post on open-source router systems',
    sourceUrl: 'https://x.com/ClementDelangue/status/2071987200481202217',
    statement: 'The future is multi-models and you’ll want to customize your router the same way you customize your code!',
  },
  {
    image: '/img/testimonials/mark-papermaster.jpeg',
    kind: 'quote',
    label: 'System-level optimization',
    name: 'Mark Papermaster',
    role: 'AMD CTO & EVP',
    sourceLabel: 'theCUBE at RAISE Summit · 2026',
    sourceUrl: 'https://www.thecube.net/events/nyse/raise-summit-26/content/Videos/9b85b557-8da2-4e68-8621-a9cec1a84674',
    statement: 'We’ve become system optimizers, still optimizing every component, but then equally looking at how you optimize how each of the pieces come together.',
  },
  {
    image: '/img/testimonials/chris-wright.jpeg',
    kind: 'perspective',
    label: 'Hybrid enterprise AI',
    name: 'Chris Wright',
    role: 'Red Hat CTO',
    sourceLabel: 'Technically Speaking / Red Hat · 2026',
    sourceUrl: 'https://www.youtube.com/watch?v=ExbMEW-Os1I&t=386s',
    statement: 'Enterprise AI needs control over cost, privacy, and policy.',
  },
  {
    image: '/img/testimonials/jensen-huang.jpeg',
    kind: 'perspective',
    label: 'Smart router',
    name: 'Jensen Huang',
    role: 'NVIDIA Co-founder, President & CEO',
    sourceLabel: 'NVIDIA Live at CES · 2026',
    sourceUrl: 'https://www.youtube.com/live/0NBILspM4c4?t=2243s',
    statement: 'A smart router sends every task to its best-fit model.',
  },
]

interface TestimonialSequenceProps {
  duplicate?: boolean
}

function revealFocusedCard(event: React.FocusEvent<HTMLAnchorElement>): void {
  const card = event.currentTarget.closest('article')
  const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches

  card?.scrollIntoView({
    behavior: prefersReducedMotion ? 'auto' : 'smooth',
    block: 'nearest',
    inline: 'center',
  })
}

function TestimonialSequence({ duplicate = false }: TestimonialSequenceProps): JSX.Element {
  return (
    <div className={styles.sequence} aria-hidden={duplicate || undefined}>
      {testimonials.map(testimonial => (
        <article
          key={testimonial.name}
          className={styles.card}
          aria-label={duplicate ? undefined : `${testimonial.name}, ${testimonial.role}`}
        >
          <div className={styles.cardHeader}>
            <img
              className={styles.portrait}
              src={testimonial.image}
              alt=""
              loading="lazy"
            />
            <div className={styles.identity}>
              <span className={styles.cardLabel}>{testimonial.label}</span>
              <span className={styles.name}>{testimonial.name}</span>
              <span className={styles.role}>{testimonial.role}</span>
            </div>
          </div>

          {testimonial.kind === 'quote'
            ? <q className={styles.statement}>{testimonial.statement}</q>
            : <p className={`${styles.statement} ${styles.perspective}`}>{testimonial.statement}</p>}

          <a
            className={styles.sourceLink}
            href={testimonial.sourceUrl}
            target="_blank"
            rel="noopener noreferrer"
            tabIndex={duplicate ? -1 : undefined}
            onFocus={revealFocusedCard}
          >
            <span>
              <Translate id="homepage.testimonials.source">Source</Translate>
            </span>
            <span className={styles.sourceName}>{testimonial.sourceLabel}</span>
            <span className={styles.sourceArrow} aria-hidden="true">↗</span>
          </a>
        </article>
      ))}
    </div>
  )
}

export default function TestimonialsRail(): JSX.Element {
  return (
    <section className={styles.section} aria-labelledby="testimonials-heading">
      <div className="site-shell-container">
        <header className={styles.header}>
          <SectionLabel>
            <Translate id="homepage.testimonials.label">Industry voices</Translate>
          </SectionLabel>
          <h2 className={styles.title} id="testimonials-heading">
            <Translate id="homepage.testimonials.title">The shift is already underway.</Translate>
          </h2>
        </header>
      </div>

      <div className={styles.viewport}>
        <div className={styles.track}>
          <TestimonialSequence />
          <TestimonialSequence duplicate />
        </div>
      </div>
    </section>
  )
}
