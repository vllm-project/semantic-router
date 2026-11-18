import React from 'react'
import Translate from '@docusaurus/Translate'
import useDocusaurusContext from '@docusaurus/useDocusaurusContext'
import Link from '@docusaurus/Link'
import { SectionLabel } from '@site/src/components/site/Chrome'
import duo from '@site/src/components/duo/duo.module.css'
import { localizeResearchEntries, researchPapers, sortResearchEntries } from '@site/src/data/researchContent'
import styles from './index.module.css'

export default function ResearchPaperCarousel(): JSX.Element {
  const { i18n } = useDocusaurusContext()
  const orderedPapers = localizeResearchEntries(
    sortResearchEntries(researchPapers),
    i18n.currentLocale,
  )

  return (
    <section className={styles.section}>
      <div className={styles.wrap}>
        <header className="site-section-intro">
          <SectionLabel>
            <Translate id="homepage.researchCarousel.label">Research</Translate>
          </SectionLabel>
          <h2>
            <Translate id="homepage.researchCarousel.title">Papers behind the router</Translate>
          </h2>
          <p>
            <Translate id="homepage.researchCarousel.subtitle">
              The ideas that move inference forward.
            </Translate>
          </p>
        </header>

        <div className={styles.carouselShell}>
          <div className={styles.carouselContainer}>
            <div className={styles.carouselTrack}>
              {orderedPapers.map((paper, index) => {
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
                        className={duo.more}
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
              Explore the research.
            </Translate>
          </p>
          <Link className={styles.all} to="/publications">
            <Translate id="homepage.researchCarousel.cta">All papers and talks</Translate>
            <span aria-hidden="true">→</span>
          </Link>
        </div>
      </div>
    </section>
  )
}
