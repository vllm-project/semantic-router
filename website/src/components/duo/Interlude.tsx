import React, { useEffect, useRef } from 'react'
import clsx from 'clsx'
import Translate, { translate } from '@docusaurus/Translate'
import { testimonials } from '@site/src/data/testimonials'
import { type InterludeKind, mountInterlude } from './interlude'
import styles from './Interlude.module.css'

function Band({ kind, seed, children }: {
  kind: InterludeKind
  seed: number
  children: React.ReactNode
}): JSX.Element {
  const ref = useRef<HTMLDivElement>(null)
  useEffect(() => (ref.current ? mountInterlude(ref.current, kind, seed) : undefined), [kind, seed])
  return (
    <div ref={ref} className={clsx(styles.band, styles[kind])}>
      <canvas className={styles.canvas} aria-hidden="true" />
      <div className={styles.content}>{children}</div>
    </div>
  )
}

// Latin words and CJK characters as separate spans, so the canvas can find each one where the browser set it.
function words(text: string): React.ReactNode[] {
  return (text.match(/[\u3000-\u9fff\uff00-\uffef]|[^\s\u3000-\u9fff\uff00-\uffef]+|\s+/g) ?? []).map((part, index) => (
    /^\s+$/.test(part) ? part : <span key={index} data-word="">{part}</span>
  ))
}

/* After Install: blocks rise off a dawn horizon into the serif line, then resolve into it. */
export function DawnInterlude(): JSX.Element {
  return (
    <Band kind="dawn" seed={3}>
      <p className={styles.dawnLine} data-line="">
        {words(translate({ id: 'homepage.interlude.dawn', message: 'Signals in. The right model out.' }))}
      </p>
    </Band>
  )
}

/* After Where it fits: a night strip of pixel stars and one row of words. */
export function NightInterlude(): JSX.Element {
  return (
    <Band kind="night" seed={7}>
      <p className={styles.nightLine}>
        <span><Translate id="homepage.interlude.night.open">Open source</Translate></span>
        <i aria-hidden="true" />
        <span><Translate id="homepage.interlude.night.hardware">Your hardware</Translate></span>
        <i aria-hidden="true" />
        <span><Translate id="homepage.interlude.night.policy">Your policy</Translate></span>
      </p>
    </Band>
  )
}

const pullQuote = testimonials.find(testimonial => testimonial.name === 'Satya Nadella') ?? testimonials[0]

/* After Research: an ember mosaic wipes across and leaves one short quote, the lead into Industry voices. */
export function EmberInterlude(): JSX.Element {
  return (
    <Band kind="ember" seed={11}>
      <figure className={styles.quote}>
        <blockquote cite={pullQuote.sourceUrl}>
          <p>{`“${pullQuote.statement}”`}</p>
        </blockquote>
        <figcaption>
          <span>{pullQuote.name}</span>
          <i aria-hidden="true" />
          <span>{pullQuote.role}</span>
          <a href={pullQuote.sourceUrl} target="_blank" rel="noopener noreferrer">
            {pullQuote.sourceLabel}
            <span aria-hidden="true"> ↗</span>
          </a>
        </figcaption>
      </figure>
    </Band>
  )
}
