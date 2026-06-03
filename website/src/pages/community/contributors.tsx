import React, { useState } from 'react'
import Layout from '@theme/Layout'
import Link from '@docusaurus/Link'
import { FaGithub } from 'react-icons/fa'
import {
  contributorRankData,
  contributorRankGeneratedAt,
  newContributorsSinceRelease,
} from '../../data/contributorRank.generated'
import type {
  ContributorRankEntry,
  ContributorRankRange,
} from '../../data/contributorRank.generated'
import styles from './contributors.module.css'

const rangeOptions: Array<{
  id: ContributorRankRange
  label: string
  caption: string
}> = [
  {
    id: 'last3months',
    label: 'Last 3 months',
    caption: 'Default',
  },
  {
    id: 'last365days',
    label: 'Last 365 days',
    caption: 'Trailing year',
  },
  {
    id: 'all',
    label: 'All time',
    caption: 'Repository history',
  },
]

const ContributorsPage: React.FC = () => {
  const [selectedRange, setSelectedRange] = useState<ContributorRankRange>('last3months')
  const snapshot = contributorRankData[selectedRange]
  const topContributors = snapshot.entries.slice(0, 5)

  return (
    <Layout
      title="Contributor Leaderboard"
      description="vLLM Semantic Router contributor leaderboard by recent and historical repository commit activity."
    >
      <main className={styles.container}>
        <header className={styles.header}>
          <div className={styles.titleBlock}>
            <p className={styles.eyebrow}>Community</p>
            <div className={styles.titleRow}>
              <h1>Contributor Leaderboard</h1>
              <Link className={styles.contributeLink} to="/community/contributing">
                Start contributing
              </Link>
            </div>
          </div>
        </header>

        <section className={styles.metrics} aria-label="Contributor rank summary">
          <Metric label="Contributors" value={snapshot.totalContributors.toLocaleString('en-US')} />
          <Metric label="New Contributors" value={newContributorsSinceRelease.totalContributors.toLocaleString('en-US')} />
          <Metric label="Commits" value={snapshot.totalCommits.toLocaleString('en-US')} />
        </section>

        <section className={styles.podiumSection} aria-label="Contributor podium">
          {topContributors.map(entry => (
            <TopContributorCard key={`${snapshot.id}-top-${entry.rank}`} entry={entry} />
          ))}
        </section>

        <section className={styles.leaderboardSection} aria-label={`${snapshot.label} contributor rank`}>
          <div className={styles.sectionHeader}>
            <div>
              <h2>{snapshot.label}</h2>
              <p>
                {formatRange(snapshot.startDate, snapshot.endDate)}
                {' '}
                · Updated
                {' '}
                {formatDate(contributorRankGeneratedAt)}
              </p>
            </div>
            <label className={styles.rangeSelectLabel}>
              <span>Range</span>
              <select
                className={styles.rangeSelect}
                value={selectedRange}
                aria-label="Contributor leaderboard time range"
                onChange={event => setSelectedRange(event.target.value as ContributorRankRange)}
              >
                {rangeOptions.map(option => (
                  <option key={option.id} value={option.id}>
                    {option.label}
                  </option>
                ))}
              </select>
            </label>
          </div>

          <div className={styles.rankListHeader} aria-hidden="true">
            <span>Rank</span>
            <span>Contributor</span>
            <span>Commits</span>
            <span>Share</span>
            <span>Latest</span>
          </div>

          <div className={styles.rankList}>
            {snapshot.entries.map(entry => (
              <ContributorRow key={`${snapshot.id}-${entry.rank}-${entry.name}`} entry={entry} />
            ))}
          </div>
        </section>
      </main>
    </Layout>
  )
}

const Metric: React.FC<{ label: string, value: string }> = ({ label, value }) => (
  <div className={styles.metric}>
    <span>{label}</span>
    <strong>{value}</strong>
  </div>
)

const TopContributorCard: React.FC<{ entry: ContributorRankEntry }> = ({ entry }) => {
  const profileUrl = entry.login ? `https://github.com/${entry.login}` : undefined

  return (
    <article className={`${styles.podiumCard} ${getPodiumClass(entry.rank)}`}>
      <div className={styles.podiumGlow} aria-hidden="true" />
      <span className={styles.podiumRank}>{formatRankNumber(entry.rank)}</span>
      <ContributorAvatar entry={entry} size="large" />
      <div className={styles.podiumIdentity}>
        <h3>{entry.name}</h3>
        {profileUrl && entry.login
          ? (
              <a href={profileUrl} target="_blank" rel="noopener noreferrer">
                <FaGithub aria-hidden="true" />
                {entry.login}
              </a>
            )
          : (
              <span>Git author</span>
            )}
      </div>
      <div className={styles.podiumStats}>
        <strong>{entry.commits.toLocaleString('en-US')}</strong>
        <span>{formatPercent(entry.share)}</span>
      </div>
    </article>
  )
}

const ContributorRow: React.FC<{ entry: ContributorRankEntry }> = ({ entry }) => {
  const profileUrl = entry.login ? `https://github.com/${entry.login}` : undefined
  const sharePercent = formatPercent(entry.share)
  const barWidth = `${Math.max(entry.share * 100, 1.5)}%`

  return (
    <article className={`${styles.rankItem} ${entry.isNewContributorSinceRelease ? styles.rankItemNew : ''}`}>
      <span className={styles.rankBadge}>
        {formatRankNumber(entry.rank)}
      </span>

      <div className={styles.contributor}>
        <ContributorAvatar entry={entry} size="compact" />
        <div className={styles.identity}>
          <span className={styles.nameLine}>
            <span className={styles.name}>{entry.name}</span>
            {entry.isNewContributorSinceRelease && (
              <span className={styles.newContributorPill}>
                New Contributor
              </span>
            )}
          </span>
          {profileUrl && entry.login
            ? (
                <a href={profileUrl} target="_blank" rel="noopener noreferrer" className={styles.githubLink}>
                  <FaGithub aria-hidden="true" />
                  {entry.login}
                </a>
              )
            : (
                <span className={styles.handle}>Git author</span>
              )}
        </div>
      </div>

      <div className={styles.statBlock}>
        <span>Commits</span>
        <strong>{entry.commits.toLocaleString('en-US')}</strong>
      </div>

      <div className={styles.share}>
        <span>{sharePercent}</span>
        <div className={styles.shareTrack} aria-hidden="true">
          <span className={styles.shareFill} style={{ width: barWidth }} />
        </div>
      </div>

      <div className={styles.statBlock}>
        <span>Latest</span>
        <strong>{formatDate(entry.latestCommitDate)}</strong>
      </div>
    </article>
  )
}

const ContributorAvatar: React.FC<{
  entry: ContributorRankEntry
  size: 'compact' | 'large'
}> = ({ entry, size }) => {
  const [didFail, setDidFail] = useState(false)
  const fallbackUrl = createFallbackAvatar(entry.avatarSeed || entry.name)
  const githubAvatarUrl = entry.avatarUrl ?? (entry.avatarLogin ? `https://github.com/${entry.avatarLogin}.png?size=160` : undefined)
  const avatarUrl = githubAvatarUrl && !didFail ? githubAvatarUrl : fallbackUrl

  return (
    <img
      className={`${styles.avatar} ${size === 'large' ? styles.avatarLarge : ''}`}
      src={avatarUrl}
      alt={`${entry.name} avatar`}
      loading="lazy"
      onError={() => setDidFail(true)}
    />
  )
}

function formatRange(startDate: string | null, endDate: string): string {
  if (!startDate) {
    return `Through ${formatDate(endDate)}`
  }

  return `${formatDate(startDate)} - ${formatDate(endDate)}`
}

function formatDate(value: string): string {
  return new Intl.DateTimeFormat('en', {
    month: 'short',
    day: 'numeric',
    year: 'numeric',
    timeZone: 'UTC',
  }).format(new Date(`${value}T00:00:00Z`))
}

function formatPercent(value: number): string {
  if (value >= 0.1) {
    return `${Math.round(value * 100)}%`
  }

  return `${(value * 100).toFixed(1)}%`
}

function getPodiumClass(rank: number): string {
  if (rank === 1) {
    return styles.podiumFirst
  }

  if (rank === 2) {
    return styles.podiumSecond
  }

  if (rank === 3) {
    return styles.podiumThird
  }

  if (rank === 4) {
    return styles.podiumFourth
  }

  return styles.podiumFifth
}

function formatRankNumber(rank: number): string {
  return `#${String(rank).padStart(2, '0')}`
}

function createFallbackAvatar(seed: string): string {
  const hash = hashString(seed)
  const hue = hash % 360
  const accent = `hsl(${hue} 38% 42%)`
  const wash = `hsl(${(hue + 24) % 360} 42% 88%)`
  const blocks = Array.from({ length: 9 }, (_, index) => {
    const x = (index % 3) * 28 + 8
    const y = Math.floor(index / 3) * 28 + 8
    const visible = (hash >> index) & 1
    const opacity = visible ? 0.72 : 0.14

    return `<rect x="${x}" y="${y}" width="20" height="20" rx="4" fill="${accent}" opacity="${opacity}"/>`
  }).join('')
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><rect width="100" height="100" rx="50" fill="${wash}"/><circle cx="72" cy="22" r="18" fill="${accent}" opacity=".12"/>${blocks}</svg>`

  return `data:image/svg+xml;charset=utf-8,${encodeURIComponent(svg)}`
}

function hashString(value: string): number {
  return [...value].reduce((hash, char) => {
    return ((hash << 5) - hash + char.charCodeAt(0)) >>> 0
  }, 2166136261)
}

export default ContributorsPage
