import React, { useState } from 'react'
import Layout from '@theme/Layout'
import Translate, { translate } from '@docusaurus/Translate'
import useDocusaurusContext from '@docusaurus/useDocusaurusContext'
import { localizeResearchEntries, researchPapers, researchTalks, sortResearchEntries } from '@site/src/data/researchContent'
import styles from './publications.module.css'

// 翻译 link label
const getLabelTranslation = (type, label) => {
  switch (type) {
    case 'paper':
      return translate({ id: 'publications.label.paper', message: 'Paper' })
    case 'event':
      if (label.includes('Watch')) {
        return translate({ id: 'publications.label.watchRecording', message: 'Watch recording' })
      }
      return translate({ id: 'publications.label.eventPage', message: 'Event page' })
    case 'video':
      return translate({ id: 'publications.label.videoRecording', message: 'Watch recording' })
    default:
      return label
  }
}

function AwardCard({ item, index }) {
  const isPaper = item.type === 'paper'
  const isFeatured = item.featured
  const isSpotlight = item.spotlight === true
  const awardTypeLabel = item.categoryLabel || (isPaper
    ? translate({ id: 'publications.awardType.paper', message: 'Research publication' })
    : translate({ id: 'publications.awardType.talk', message: 'Conference presentation' }))
  const peopleLabel = isPaper
    ? translate({ id: 'publications.detail.authors', message: 'Authors:' })
    : translate({ id: 'publications.detail.speakers', message: 'Speakers:' })
  const venueLabel = translate({ id: 'publications.detail.venue', message: 'Venue:' })

  return (
    <div
      className={`${styles.awardCard} ${isPaper ? styles.paperAward : styles.talkAward} ${isFeatured ? styles.featuredAward : ''} ${isSpotlight ? styles.spotlightCard : ''}`}
      style={{ '--animation-delay': `${index * 0.1}s` }}
    >
      {/* Award Frame */}
      <div className={styles.awardFrame}>
        {/* Award Header with Medal */}
        <div className={styles.awardHeader}>
          <div className={styles.medalContainer}>
            <div className={`${styles.medal} ${isPaper ? styles.paperMedal : styles.talkMedal}`}>
              {isPaper ? 'P' : 'T'}
            </div>
          </div>
          <div className={styles.awardType}>
            {awardTypeLabel}
          </div>
        </div>

        {/* Award Content */}
        <div className={styles.awardContent}>
          {item.categoryLabel && (
            <div className={isSpotlight ? styles.spotlightBadge : styles.categoryBadge}>
              {item.categoryLabel}
            </div>
          )}
          <h3 className={styles.awardTitle}>{item.title}</h3>

          <div className={styles.awardDetails}>
            <div className={styles.awardAuthors}>
              <span className={styles.authorLabel}>{peopleLabel}</span>
              <span className={styles.authorNames}>
                {isPaper ? item.authors : item.speakers}
              </span>
            </div>

            {item.venue && (
              <div className={styles.awardVenue}>
                <span className={styles.venueLabel}>{venueLabel}</span>
                <span className={styles.venueName}>{item.venue}</span>
              </div>
            )}
          </div>

          <div className={styles.awardDescription}>
            {item.abstract}
          </div>
        </div>

        {/* Year and Actions Row */}
        <div className={styles.yearAndActions}>
          <span className={styles.awardYear}>{item.year}</span>
          {item.links && item.links.length > 0 && (
            <>
              {item.links.map((link, linkIndex) => (
                <a
                  key={linkIndex}
                  href={link.url}
                  className={`${styles.awardLink} ${
                    link.type === 'paper' ? styles.primaryLink : styles.secondaryLink
                  }`}
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  {getLabelTranslation(link.type, link.label)}
                </a>
              ))}
            </>
          )}
        </div>

        {/* Award Footer */}
        <div className={styles.awardFooter}>
          <div className={styles.awardSeal}>
            <div className={styles.sealInner}>
              <span className={styles.sealText}>vLLM</span>
              <span className={styles.sealSubtext}>Semantic Router</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default function Publications() {
  const { i18n } = useDocusaurusContext()
  const [activeFilter, setActiveFilter] = useState('all')

  const sortedPapers = localizeResearchEntries(
    sortResearchEntries(researchPapers),
    i18n.currentLocale,
  )
  const sortedTalks = localizeResearchEntries(
    sortResearchEntries(researchTalks),
    i18n.currentLocale,
  )

  const allItems = [...sortedPapers, ...sortedTalks]
  const filteredItems = activeFilter === 'all'
    ? allItems
    : allItems.filter(item => item.type === activeFilter)

  const paperCount = researchPapers.length
  const talkCount = researchTalks.length
  const totalCount = paperCount + talkCount
  const emptyMessage = activeFilter === 'all'
    ? translate({ id: 'publications.empty.all', message: 'No items found.' })
    : activeFilter === 'paper'
      ? translate({ id: 'publications.empty.paper', message: 'No papers found.' })
      : translate({ id: 'publications.empty.talk', message: 'No talks found.' })

  return (
    <Layout
      title={translate({ id: 'publications.title', message: 'Papers & Talks' })}
      description={translate({
        id: 'publications.meta.description',
        message: 'Latest research publications, talks, and scientific contributions from the vLLM Semantic Router project.',
      })}
    >
      <div className={`${styles.container} publications-page`}>
        <header className={styles.header}>
          <div className={styles.wallDecoration}>
            <div className={styles.wallPattern}></div>
          </div>
          <h1 className={styles.title}>
            <Translate id="publications.title">Papers & Talks</Translate>
          </h1>
          <p className={styles.subtitle}>
            <span className={styles.subtitleHighlight}>
              <Translate id="publications.subtitle">Research, talks, and position papers from the vLLM Semantic Router project.</Translate>
            </span>
          </p>
        </header>

        <div className={styles.filterSection}>
          <div className={styles.filterButtons}>
            <button
              className={`${styles.filterButton} ${activeFilter === 'all' ? styles.active : ''}`}
              onClick={() => setActiveFilter('all')}
            >
              <Translate id="publications.filter.all">All</Translate>
              {' '}
              (
              {totalCount}
              )
            </button>
            <button
              className={`${styles.filterButton} ${activeFilter === 'paper' ? styles.active : ''}`}
              onClick={() => setActiveFilter('paper')}
            >
              <Translate id="publications.filter.papers">Papers</Translate>
              {' '}
              (
              {paperCount}
              )
            </button>
            <button
              className={`${styles.filterButton} ${activeFilter === 'talk' ? styles.active : ''}`}
              onClick={() => setActiveFilter('talk')}
            >
              <Translate id="publications.filter.talks">Talks</Translate>
              {' '}
              (
              {talkCount}
              )
            </button>
          </div>
        </div>

        <main>
          {activeFilter === 'all'
            ? (
                <div className={styles.awardWall}>
                  {/* Research Publications Wall */}
                  <section className={styles.awardSection}>
                    <div className={styles.sectionHeader}>
                      <h2 className={styles.sectionTitle}>
                        <Translate id="publications.papers.title">Research Publications</Translate>
                      </h2>
                      <div className={styles.sectionDivider}></div>
                    </div>
                    <div className={styles.awardsGrid}>
                      {sortedPapers.map((item, index) => (
                        <AwardCard key={item.id} item={item} index={index} />
                      ))}
                    </div>
                  </section>

                  {/* Conference Presentations Wall */}
                  <section className={styles.awardSection}>
                    <div className={styles.sectionHeader}>
                      <h2 className={styles.sectionTitle}>
                        <Translate id="publications.talks.title">Conference Presentations</Translate>
                      </h2>
                      <div className={styles.sectionDivider}></div>
                    </div>
                    <div className={styles.awardsGrid}>
                      {sortedTalks.map((item, index) => (
                        <AwardCard key={item.id} item={item} index={index + sortedPapers.length} />
                      ))}
                    </div>
                  </section>
                </div>
              )
            : (
                <div className={styles.filteredAwards}>
                  <div className={styles.awardsGrid}>
                    {filteredItems.map((item, index) => (
                      <AwardCard key={item.id} item={item} index={index} />
                    ))}
                  </div>
                </div>
              )}

          {filteredItems.length === 0 && (
            <div className={styles.emptyState}>
              <p>{emptyMessage}</p>
            </div>
          )}
        </main>
      </div>
    </Layout>
  )
}
