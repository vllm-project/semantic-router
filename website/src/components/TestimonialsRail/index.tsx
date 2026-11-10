import React, { useEffect, useRef, useState } from 'react'
import clsx from 'clsx'
import Translate, { translate } from '@docusaurus/Translate'
import ThemedImage from '@theme/ThemedImage'
import { SectionLabel } from '@site/src/components/site/Chrome'
import { testimonials } from '@site/src/data/testimonials'
import styles from './index.module.css'

// The pull quote above this section is Satya Nadella's, so his slide comes last rather than first.
const slides = [...testimonials.slice(1), testimonials[0]]
const COUNT = slides.length
const INTERVAL = 7000

/* Industry voices as a carousel: one voice at a time on a card sized to the tallest quote, advancing slowly. It pauses
 * on hover, focus, the Pause button, a hidden tab and off screen, and never moves by itself under reduced motion.
 * Prev and next, the dots, a swipe and the arrow keys move it; a polite status line says which slide is shown. */
export default function TestimonialsRail(): JSX.Element {
  const regionRef = useRef<HTMLDivElement>(null)
  const swipe = useRef<{ x: number, y: number } | null>(null)
  const [index, setIndex] = useState(0)
  const [announce, setAnnounce] = useState(false)
  const [paused, setPaused] = useState(false)
  const [hovered, setHovered] = useState(false)
  const [focused, setFocused] = useState(false)
  const [onscreen, setOnscreen] = useState(false)
  const [tabHidden, setTabHidden] = useState(false)
  const [still, setStill] = useState(true)

  useEffect(() => {
    const query = window.matchMedia('(prefers-reduced-motion: reduce)')
    const update = () => setStill(query.matches)
    update()
    query.addEventListener('change', update)
    const onVisibility = () => setTabHidden(document.hidden)
    document.addEventListener('visibilitychange', onVisibility)
    const element = regionRef.current
    const observer = element
      ? new IntersectionObserver(entries => setOnscreen(Boolean(entries[0]?.isIntersecting)), { threshold: 0.3 })
      : undefined
    if (element) observer?.observe(element)
    return () => {
      query.removeEventListener('change', update)
      document.removeEventListener('visibilitychange', onVisibility)
      observer?.disconnect()
    }
  }, [])

  const rotating = !paused && !hovered && !focused && onscreen && !tabHidden && !still

  useEffect(() => {
    if (!rotating) return undefined
    const timer = window.setTimeout(() => {
      setAnnounce(false)
      setIndex(current => (current + 1) % COUNT)
    }, INTERVAL)
    return () => window.clearTimeout(timer)
  }, [rotating, index])

  const go = (next: number) => {
    setAnnounce(true)
    setIndex((next + COUNT) % COUNT)
  }

  const onKeyDown = (event: React.KeyboardEvent) => {
    if (event.key === 'ArrowLeft') go(index - 1)
    else if (event.key === 'ArrowRight') go(index + 1)
    else return
    event.preventDefault()
  }

  const onPointerDown = (event: React.PointerEvent) => {
    if (event.pointerType !== 'mouse') swipe.current = { x: event.clientX, y: event.clientY }
  }
  const onPointerUp = (event: React.PointerEvent) => {
    const start = swipe.current
    swipe.current = null
    if (!start) return
    const dx = event.clientX - start.x
    if (Math.abs(dx) > 40 && Math.abs(dx) > Math.abs(event.clientY - start.y)) go(index + (dx < 0 ? 1 : -1))
  }

  const toggleLabel = paused
    ? translate({ id: 'homepage.testimonials.play', message: 'Play the voices' })
    : translate({ id: 'homepage.testimonials.pause', message: 'Pause the voices' })

  return (
    <section className={styles.section} aria-labelledby="testimonials-heading">
      <div className={styles.wrap}>
        <header className="site-section-intro">
          <SectionLabel>
            <Translate id="homepage.testimonials.label">Industry voices</Translate>
          </SectionLabel>
          <h2 id="testimonials-heading">
            <Translate id="homepage.testimonials.title">The shift is already underway</Translate>
          </h2>
        </header>

        <div
          ref={regionRef}
          className={styles.carousel}
          role="region"
          aria-roledescription={translate({ id: 'homepage.testimonials.roleCarousel', message: 'carousel' })}
          aria-labelledby="testimonials-heading"
          onKeyDown={onKeyDown}
          onPointerEnter={event => event.pointerType === 'mouse' && setHovered(true)}
          onPointerLeave={event => event.pointerType === 'mouse' && setHovered(false)}
          onFocus={() => setFocused(true)}
          onBlur={(event) => {
            if (!event.currentTarget.contains(event.relatedTarget as Node | null)) setFocused(false)
          }}
        >
          <div className={styles.stage} onPointerDown={onPointerDown} onPointerUp={onPointerUp} onPointerCancel={() => (swipe.current = null)}>
            {slides.map((voice, slide) => {
              const active = slide === index
              return (
                <article
                  key={voice.name}
                  className={clsx(styles.slide, active && styles.active)}
                  role="group"
                  aria-roledescription={translate({ id: 'homepage.testimonials.roleSlide', message: 'slide' })}
                  aria-label={translate(
                    { id: 'homepage.testimonials.slideLabel', message: '{n} of {count}: {name}' },
                    { n: slide + 1, count: COUNT, name: voice.name },
                  )}
                  aria-hidden={active ? undefined : true}
                >
                  <span className={styles.cardLabel}>{voice.label}</span>
                  {voice.kind === 'quote'
                    ? <q className={styles.statement}>{voice.statement}</q>
                    : <p className={clsx(styles.statement, styles.perspective)}>{voice.statement}</p>}
                  <footer className={styles.person}>
                    <img className={styles.portrait} src={voice.image} alt="" width={56} height={56} loading="lazy" decoding="async" />
                    <span className={styles.identity}>
                      <span className={styles.name}>{voice.name}</span>
                      <span className={styles.role}>{voice.role}</span>
                    </span>
                    <ThemedImage className={styles.logo} alt={voice.logo.alt} sources={{ light: voice.logo.light, dark: voice.logo.dark }} />
                    <a
                      className={styles.sourceLink}
                      href={voice.sourceUrl}
                      target="_blank"
                      rel="noopener noreferrer"
                      tabIndex={active ? undefined : -1}
                    >
                      <Translate id="homepage.testimonials.source">Source</Translate>
                      <span className={styles.sourceName}>{voice.sourceLabel}</span>
                      <span className={styles.sourceArrow} aria-hidden="true">↗</span>
                    </a>
                  </footer>
                </article>
              )
            })}
          </div>

          <div className={styles.controls}>
            <button type="button" className={styles.toggle} onClick={() => setPaused(value => !value)} aria-label={toggleLabel} title={toggleLabel}>
              <svg viewBox="0 0 8 8" aria-hidden="true" shapeRendering="crispEdges">
                {paused
                  ? <path d="M2 1h1v6H2zM3 2h1v4H3zM4 3h1v2H4z" fill="currentColor" />
                  : <path d="M2 1h1v6H2zM5 1h1v6H5z" fill="currentColor" />}
              </svg>
            </button>
            <button
              type="button"
              className={styles.arrow}
              onClick={() => go(index - 1)}
              aria-label={translate({ id: 'homepage.testimonials.prev', message: 'Previous voice' })}
            >
              <span aria-hidden="true">‹</span>
            </button>
            <div className={styles.dots}>
              {slides.map((voice, slide) => (
                <button
                  key={voice.name}
                  type="button"
                  className={clsx(styles.dot, slide === index && styles.dotActive)}
                  onClick={() => go(slide)}
                  aria-label={translate(
                    { id: 'homepage.testimonials.dot', message: 'Show voice {n}: {name}' },
                    { n: slide + 1, name: voice.name },
                  )}
                  aria-current={slide === index ? 'true' : undefined}
                />
              ))}
            </div>
            <button
              type="button"
              className={styles.arrow}
              onClick={() => go(index + 1)}
              aria-label={translate({ id: 'homepage.testimonials.next', message: 'Next voice' })}
            >
              <span aria-hidden="true">›</span>
            </button>
          </div>

          <p className={styles.srOnly} aria-live="polite" aria-atomic="true">
            {announce
              ? translate(
                  { id: 'homepage.testimonials.status', message: 'Slide {n} of {count}' },
                  { n: index + 1, count: COUNT },
                )
              : ''}
          </p>
        </div>
      </div>
    </section>
  )
}
