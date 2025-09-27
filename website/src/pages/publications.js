import React from 'react'
import Layout from '@theme/Layout'
import styles from './publications.module.css'

const publications = [
  {
    id: 1,
    title: 'When to Reason: Semantic Router for vLLM',
    authors: 'Chen Wang, Xunzhuo Liu, Yuhan Liu, Yue Zhu, Xiangxi Mo, Junchen Jiang, Huamin Chen',
    venue: 'NeurIPS - MLForSys',
    year: '2025',
    abstract: 'We propose vLLM semantic router integrated with vLLM that selectively applies reasoning only when beneficial, achieving over 10 percentage point accuracy gains while nearly halving latency and token usage',
    links: [
      { type: 'paper', url: 'https://mlforsystems.org', label: '📄 Paper' },
    ],
    featured: true,
  },
]

function PublicationCard({ publication }) {
  return (
    <div className={styles.publicationCard}>
      <h3 className={styles.paperTitle}>{publication.title}</h3>
      <p className={styles.paperAuthors}>{publication.authors}</p>
      <span className={styles.paperVenue}>
        {publication.venue}
        {' '}
        {publication.year}
      </span>
      <p className={styles.paperAbstract}>{publication.abstract}</p>
      <div className={styles.paperLinks}>
        {publication.links.map((link, index) => (
          <a
            key={index}
            href={link.url}
            className={`${styles.paperLink} ${
              link.type === 'paper' ? styles.paperLinkPrimary : styles.paperLinkSecondary
            }`}
            target="_blank"
            rel="noopener noreferrer"
          >
            {link.label}
          </a>
        ))}
      </div>
    </div>
  )
}

export default function Publications() {
  return (
    <Layout
      title="Publications"
      description="Latest research publications and scientific contributions from the vLLM Semantic Router project"
    >
      <div className={styles.container}>
        <header className={styles.header}>
          <h1 className={styles.title}>🎓 Publications</h1>
          <p className={styles.subtitle}>
            Discover community-driven latest research contributions in LLM and intelligent routing systems.
            Our work pushes the boundaries of efficient LLM inference.
          </p>
        </header>

        <main>
          <div className={styles.publicationsList}>
            {publications.map(publication => (
              <PublicationCard key={publication.id} publication={publication} />
            ))}
          </div>
        </main>
      </div>
    </Layout>
  )
}
