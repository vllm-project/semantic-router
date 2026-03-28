import React, { useEffect, useRef, useState } from 'react'
import Translate from '@docusaurus/Translate'
import useDocusaurusContext from '@docusaurus/useDocusaurusContext'
import { PillLink, SectionLabel } from '@site/src/components/site/Chrome'
import { localizeResearchEntries, researchPapers, sortResearchEntries } from '@site/src/data/researchContent'
import styles from './index.module.css'

export default function ResearchPaperCarousel(): JSX.Element {
  const { i18n } = useDocusaurusContext()
  const trackRef = useRef<HTMLDivElement>(null)
  const [autoScrollEnabled, setAutoScrollEnabled] = useState(false)
  const orderedPapers = localizeResearchEntries(
    sortResearchEntries(researchPapers),
    i18n.currentLocale,
  )

  useEffect(() => {
    if (typeof window === 'undefined') {
      return undefined
    }

    const mediaQuery = window.matchMedia('(prefers-reduced-motion: reduce)')
    const syncPreference = () => {
      setAutoScrollEnabled(!mediaQuery.matches)
    }

    syncPreference()

    if (typeof mediaQuery.addEventListener === 'function') {
      mediaQuery.addEventListener('change', syncPreference)
      return () => {
        mediaQuery.removeEventListener('change', syncPreference)
      }
    }

    mediaQuery.addListener(syncPreference)
    return () => {
      mediaQuery.removeListener(syncPreference)
    }
  }, [])

  useEffect(() => {
    const track = trackRef.current
    if (!track) {
      return undefined
    }

    if (!autoScrollEnabled) {
      track.style.transform = ''
      return undefined
    }

    let animationFrameId = 0
    let scrollPosition = 0
    let totalWidth = 0
    let paused = false
    const scrollSpeed = 0.35

    const updateTotalWidth = () => {
      const cards = Array.from(track.children).slice(0, orderedPapers.length)
      const gap = parseFloat(window.getComputedStyle(track).gap || '0')

      totalWidth = cards.reduce((total, card, index) => {
        const width = (card as HTMLElement).offsetWidth
        return total + width + (index < cards.length - 1 ? gap : 0)
      }, 0)
    }

    const pause = () => {
      paused = true
    }

    const resume = () => {
      paused = false
    }

    const scroll = () => {
      if (!paused && totalWidth > 0) {
        scrollPosition += scrollSpeed

        if (scrollPosition >= totalWidth) {
          scrollPosition = 0
        }

        track.style.transform = `translateX(-${scrollPosition}px)`
      }

      animationFrameId = window.requestAnimationFrame(scroll)
    }

    updateTotalWidth()
    window.addEventListener('resize', updateTotalWidth)
    track.addEventListener('mouseenter', pause)
    track.addEventListener('mouseleave', resume)
    track.addEventListener('focusin', pause)
    track.addEventListener('focusout', resume)
    animationFrameId = window.requestAnimationFrame(scroll)

    return () => {
      window.cancelAnimationFrame(animationFrameId)
      window.removeEventListener('resize', updateTotalWidth)
      track.removeEventListener('mouseenter', pause)
      track.removeEventListener('mouseleave', resume)
      track.removeEventListener('focusin', pause)
      track.removeEventListener('focusout', resume)
      track.style.transform = ''
    }
  }, [autoScrollEnabled])

  const displayPapers = autoScrollEnabled
    ? [...orderedPapers, ...orderedPapers, ...orderedPapers]
    : orderedPapers

  return (
    <section className={styles.section}>
      <div className="site-shell-container">
        <div className={styles.header}>
          <div>
            <SectionLabel>
              <Translate id="homepage.researchCarousel.label">Research</Translate>
            </SectionLabel>
            <h2 className={styles.title}>
              <Translate id="homepage.researchCarousel.title">Papers behind the router.</Translate>
            </h2>
          </div>
          <p className={styles.subtitle}>
            <Translate id="homepage.researchCarousel.subtitle">
              Research threads that trace the router&apos;s evolving ideas across safety,
              multimodality, orchestration, and system design.
            </Translate>
          </p>
        </div>

        <div className={styles.carouselShell}>
          <div className={`${styles.carouselContainer} ${!autoScrollEnabled ? styles.carouselContainerStatic : ''}`}>
            <div className={styles.carouselTrack} ref={trackRef}>
              {displayPapers.map((paper, index) => {
                const paperLink = paper.links.find(link => link.type === 'paper') ?? paper.links[0]
                return (
                  <article
                    key={`${paper.id}-${index}`}
                    className={`${styles.paperCard} ${paper.spotlight ? styles.spotlightCard : ''}`}
                  >
                    <div className={styles.cardTop}>
                      <span className={styles.cardMeta}>
                        {paper.year}
                        {' '}
                        /
                        {' '}
                        <Translate id="homepage.researchCarousel.cardType">Paper</Translate>
                      </span>
                      {paper.categoryLabel && (
                        <span className={styles.categoryBadge}>{paper.categoryLabel}</span>
                      )}
                    </div>

                    <div className={styles.cardCopy}>
                      <h3 className={styles.paperTitle}>{paper.title}</h3>
                      <p className={styles.paperAuthors}>{paper.authors}</p>
                      {paper.venue && <p className={styles.paperVenue}>{paper.venue}</p>}
                      <p className={styles.paperAbstract}>{paper.abstract}</p>
                    </div>

                    <div className={styles.cardFooter}>
                      <a
                        href={paperLink.url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className={styles.paperLink}
                      >
                        <Translate id="homepage.researchCarousel.paperLink">Read paper</Translate>
                      </a>
                    </div>
                  </article>
                )
              })}
            </div>
          </div>
        </div>

        <div className={styles.footer}>
          <p>
            <Translate id="homepage.researchCarousel.footer">
              Papers that frame how the router sees, decides, and scales.
            </Translate>
          </p>
          <PillLink to="/publications" muted>
            <Translate id="homepage.researchCarousel.cta">See all papers and talks</Translate>
          </PillLink>
        </div>
      </div>
    </section>
  )
}
