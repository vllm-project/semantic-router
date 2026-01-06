import React, { useState } from 'react'
import Layout from '@theme/Layout'
import Translate, { translate } from '@docusaurus/Translate'
import styles from './publications.module.css'

// 翻译 link label
const getLabelTranslation = (type, label) => {
  switch (type) {
    case 'paper':
      return translate({ id: 'publications.label.paper', message: '📄 Paper' })
    case 'event':
      if (label.includes('Watch')) {
        return translate({ id: 'publications.label.watchRecording', message: '🎤 Watch Recording' })
      }
      return translate({ id: 'publications.label.eventPage', message: '🎤 Event Page' })
    case 'video':
      return translate({ id: 'publications.label.videoRecording', message: '📹 Watch Recording' })
    default:
      return label
  }
}

const papers = [
  {
    id: 1,
    type: 'paper',
    title: 'When to Reason: Semantic Router for vLLM',
    authors: 'Chen Wang, Xunzhuo Liu, Yuhan Liu, Yue Zhu, Xiangxi Mo, Junchen Jiang, Huamin Chen',
    venue: 'NeurIPS - MLForSys',
    year: '2025',
    abstract: 'We present a semantic router that classifies queries based on their reasoning requirements and selectively applies reasoning only when beneficial.',
    links: [
      { type: 'paper', url: 'https://arxiv.org/abs/2510.08731', label: '📄 Paper' },
    ],
    featured: true,
  },
  {
    id: 2,
    type: 'paper',
    title: 'Category-Aware Semantic Caching for Heterogeneous LLM Workloads',
    authors: 'Chen Wang, Xunzhuo Liu, Yue Zhu, Alaa Youssef, Priya Nagpurkar, Huamin Chen',
    venue: '',
    year: '2025',
    abstract: 'We present a category-aware semantic caching where similarity thresholds, TTLs, and quotas vary by query category, with a hybrid architecture separating in-memory HNSW search from external document storage.',
    links: [
      { type: 'paper', url: 'https://arxiv.org/abs/2510.26835', label: '📄 Paper' },
    ],
    featured: true,
  },
  {
    id: 3,
    type: 'paper',
    title: 'Semantic Inference Routing Protocol (SIRP)',
    authors: 'Huamin Chen, Luay Jalil',
    venue: 'Internet Engineering Task Force (IETF)',
    year: '2025',
    abstract: 'This document specifies the Semantic Inference Routing Protocol (SIRP), a framework for content-level classification and semantic routing in AI inference systems. ',
    links: [
      { type: 'paper', url: 'https://datatracker.ietf.org/doc/html/draft-chen-nmrg-semantic-inference-routing', label: '📄 Paper' },
    ],
    featured: true,
  },
  {
    id: 4,
    type: 'paper',
    title: 'Multi-Provider Extensions for Agentic AI Inference APIs',
    authors: 'H. Chen, L. Jalil, N. Cocker',
    venue: 'Internet Engineering Task Force (IETF) - Network Management Research Group',
    year: '2025',
    abstract: 'This document specifies multi-provider extensions for agentic AI inference APIs. Published: 20 October 2025. Intended Status: Informational. Expires: 23 April 2026.',
    links: [
      { type: 'paper', url: 'https://www.ietf.org/archive/id/draft-chen-nmrg-multi-provider-inference-api-00.html', label: '📄 Paper' },
    ],
    featured: true,
  },
]

const talks = [
  {
    id: 4,
    type: 'talk',
    title: 'Intelligent LLM Routing: A New Paradigm for Multi-Model AI Orchestration in Kubernetes',
    speakers: 'Chen Wang, Huamin Chen',
    venue: 'KubeCon NA 2025',
    organization: '',
    year: '2025',
    abstract: 'This research-driven talk introduces a novel architecture paradigm that complements recent advances in timely intelligent inference routing for large language models.',
    links: [
      { type: 'event', url: 'https://kccncna2025.sched.com/event/27FaI?iframe=no', label: '🎤 Event Page' },
    ],
    featured: true,
  },
  {
    id: 5,
    type: 'talk',
    title: 'vLLM Semantic Router: Unlock the Power of Intelligent Routing',
    speakers: 'Xunzhuo Liu',
    venue: 'vLLM Meetup Beijing',
    organization: '',
    year: '2025',
    abstract: 'A deep dive into vLLM Semantic Router capabilities, demonstrating how intelligent routing can unlock new possibilities for efficient LLM inference.',
    links: [
      { type: 'event', url: 'https://drive.google.com/drive/folders/1nQJ8ZkLSjKxvu36sSHaceVXtttbLvvu-', label: '🎤 Watch Recording' },
    ],
    featured: true,
  },
  {
    id: 6,
    type: 'talk',
    title: 'AI-Powered vLLM Semantic Router',
    speakers: 'Huamin Chen',
    venue: 'vLLM Office Hours',
    organization: '',
    year: '2025',
    abstract: 'An overview of AI-powered features in vLLM Semantic Router, showcasing the latest developments and community contributions.',
    links: [
      { type: 'video', url: 'https://www.youtube.com/live/b-ciRqvbtsk', label: '📹 Watch Recording' },
    ],
    featured: true,
  },
]

function AwardCard({ item, index }) {
  const isPaper = item.type === 'paper'
  const isFeatured = item.featured

  return (
    <div
      className={`${styles.awardCard} ${isPaper ? styles.paperAward : styles.talkAward} ${isFeatured ? styles.featuredAward : ''}`}
      style={{ '--animation-delay': `${index * 0.1}s` }}
    >
      {/* Award Frame */}
      <div className={styles.awardFrame}>
        {/* Award Header with Medal */}
        <div className={styles.awardHeader}>
          <div className={styles.medalContainer}>
            <div className={`${styles.medal} ${isPaper ? styles.paperMedal : styles.talkMedal}`}>
              {isPaper ? '🏆' : '🤗'}
            </div>
            {isFeatured && <div className={styles.starBadge}>✨</div>}
          </div>
          <div className={styles.awardType}>
            {isPaper ? 'RESEARCH PUBLICATION' : 'CONFERENCE PRESENTATION'}
          </div>
        </div>

        {/* Award Content */}
        <div className={styles.awardContent}>
          <h3 className={styles.awardTitle}>{item.title}</h3>

          <div className={styles.awardDetails}>
            <div className={styles.awardAuthors}>
              <span className={styles.authorLabel}>
                {isPaper ? 'Authors:' : 'Speakers:'}
              </span>
              <span className={styles.authorNames}>
                {isPaper ? item.authors : item.speakers}
              </span>
            </div>

            {item.venue && (
              <div className={styles.awardVenue}>
                <span className={styles.venueLabel}>Venue:</span>
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
  const [activeFilter, setActiveFilter] = useState('all')

  const sortedPapers = papers.sort((a, b) => {
    // Sort by featured first, then by year (descending), then by id
    if (a.featured && !b.featured) return -1
    if (!a.featured && b.featured) return 1
    if (a.year !== b.year) return parseInt(b.year) - parseInt(a.year)
    return a.id - b.id
  })

  const sortedTalks = talks.sort((a, b) => {
    // Sort by featured first, then by year (descending), then by id
    if (a.featured && !b.featured) return -1
    if (!a.featured && b.featured) return 1
    if (a.year !== b.year) return parseInt(b.year) - parseInt(a.year)
    return a.id - b.id
  })

  const allItems = [...sortedPapers, ...sortedTalks]
  const filteredItems = activeFilter === 'all'
    ? allItems
    : allItems.filter(item => item.type === activeFilter)

  const paperCount = papers.length
  const talkCount = talks.length
  const totalCount = paperCount + talkCount

  return (
    <Layout
      title="Papers & Talks"
      description="Latest research publications, talks, and scientific contributions from the vLLM Semantic Router project"
    >
      <div className={`${styles.container} publications-page`}>
        <header className={styles.header}>
          <div className={styles.wallDecoration}>
            <div className={styles.wallPattern}></div>
          </div>
          <h1 className={styles.title}>
            🏆
            <Translate id="publications.title">Papers & Talks</Translate>
          </h1>
          <p className={styles.subtitle}>
            <span className={styles.subtitleHighlight}>
              <Translate id="publications.subtitle">Innovation thrives when great minds come together ❤️</Translate>
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
              <Translate id="publications.filter.papers">📄 Papers</Translate>
              {' '}
              (
              {paperCount}
              )
            </button>
            <button
              className={`${styles.filterButton} ${activeFilter === 'talk' ? styles.active : ''}`}
              onClick={() => setActiveFilter('talk')}
            >
              <Translate id="publications.filter.talks">🎤 Talks</Translate>
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
                        🏆
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
                        🏆
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
              <p>
                No
                {activeFilter === 'all' ? 'items' : activeFilter + 's'}
                {' '}
                found.
              </p>
            </div>
          )}
        </main>
      </div>
    </Layout>
  )
}
