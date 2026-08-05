import React, { useState } from 'react'
import Layout from '@theme/Layout'
import Translate, { translate } from '@docusaurus/Translate'
import useDocusaurusContext from '@docusaurus/useDocusaurusContext'
import ResearchLayout from '@site/src/components/research/ResearchLayout'
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

function ResearchEntryRow({ item }) {
  const isPaper = item.type === 'paper'
  const typeLabel = isPaper
    ? translate({ id: 'publications.type.paper', message: 'Paper' })
    : translate({ id: 'publications.type.talk', message: 'Talk' })
  const people = isPaper ? item.authors : item.speakers

  return (
    <article className={styles.row}>
      <span className={styles.rowYear}>{item.year}</span>
      <span className={`${styles.rowType} ${isPaper ? styles.rowTypePaper : styles.rowTypeTalk}`}>
        {typeLabel}
      </span>
      <div className={styles.rowBody}>
        <h3 className={styles.rowTitle}>{item.title}</h3>
        <p className={styles.rowMeta}>
          {people}
          {item.venue && (
            <>
              {' · '}
              {item.venue}
            </>
          )}
        </p>
      </div>
      {item.links && item.links.length > 0 && (
        <div className={styles.rowLinks}>
          {item.links.map((link, linkIndex) => (
            <a
              key={linkIndex}
              href={link.url}
              className={styles.rowLink}
              target="_blank"
              rel="noopener noreferrer"
            >
              {getLabelTranslation(link.type, link.label)}
              <span aria-hidden="true">↗</span>
            </a>
          ))}
        </div>
      )}
    </article>
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
      <ResearchLayout
        activeKey="publications"
        title={<Translate id="publications.title">Papers & Talks</Translate>}
        description={<Translate id="publications.subtitle">Research, talks, and position papers from the vLLM Semantic Router project.</Translate>}
      >
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

        {activeFilter === 'all'
          ? (
              <div className={styles.groups}>
                <section className={styles.group}>
                  <h2 className={styles.groupTitle}>
                    <Translate id="publications.papers.title">Research Publications</Translate>
                  </h2>
                  <div className={styles.list}>
                    {sortedPapers.map(item => (
                      <ResearchEntryRow key={item.id} item={item} />
                    ))}
                  </div>
                </section>

                <section className={styles.group}>
                  <h2 className={styles.groupTitle}>
                    <Translate id="publications.talks.title">Conference Presentations</Translate>
                  </h2>
                  <div className={styles.list}>
                    {sortedTalks.map(item => (
                      <ResearchEntryRow key={item.id} item={item} />
                    ))}
                  </div>
                </section>
              </div>
            )
          : (
              <div className={styles.list}>
                {filteredItems.map(item => (
                  <ResearchEntryRow key={item.id} item={item} />
                ))}
              </div>
            )}

        {filteredItems.length === 0 && (
          <div className={styles.emptyState}>
            <p>{emptyMessage}</p>
          </div>
        )}
      </ResearchLayout>
    </Layout>
  )
}
