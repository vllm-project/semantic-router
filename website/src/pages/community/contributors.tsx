import React, { useState } from 'react'
import Layout from '@theme/Layout'
import Link from '@docusaurus/Link'
import Translate, { translate } from '@docusaurus/Translate'
import useDocusaurusContext from '@docusaurus/useDocusaurusContext'
import { FaGithub } from 'react-icons/fa'
import {
  contributorRankData,
  contributorRankGeneratedAt,
} from '../../data/contributorRank.generated'
import type {
  ContributorRankEntry,
  ContributorRankRange,
} from '../../data/contributorRank.generated'
import styles from './contributors.module.css'

type RangeOption = {
  id: ContributorRankRange
  label: string
  caption: string
}

const ContributorsPage: React.FC = () => {
  const { i18n } = useDocusaurusContext()
  const currentLocale = i18n.currentLocale
  const numberLocale = currentLocale === 'zh-Hans' ? 'zh-CN' : 'en-US'
  const dateLocale = currentLocale === 'zh-Hans' ? 'zh-CN' : 'en'

  const rangeOptions: RangeOption[] = [
    {
      id: 'v03ToNow',
      label: translate({
        id: 'community.contributors.range.v03ToNow.label',
        message: 'v0.3 -> Now',
      }),
      caption: translate({
        id: 'community.contributors.range.v03ToNow.caption',
        message: 'Current release',
      }),
    },
    {
      id: 'v02ToV03',
      label: translate({
        id: 'community.contributors.range.v02ToV03.label',
        message: 'v0.2 -> v0.3',
      }),
      caption: translate({
        id: 'community.contributors.range.v02ToV03.caption',
        message: 'Release gap',
      }),
    },
    {
      id: 'v01ToV02',
      label: translate({
        id: 'community.contributors.range.v01ToV02.label',
        message: 'v0.1 -> v0.2',
      }),
      caption: translate({
        id: 'community.contributors.range.v01ToV02.caption',
        message: 'Release gap',
      }),
    },
    {
      id: 'v0ToV01',
      label: translate({
        id: 'community.contributors.range.v0ToV01.label',
        message: 'v0 -> v0.1',
      }),
      caption: translate({
        id: 'community.contributors.range.v0ToV01.caption',
        message: 'Initial release',
      }),
    },
    {
      id: 'all',
      label: translate({
        id: 'community.contributors.range.all.label',
        message: 'All time',
      }),
      caption: translate({
        id: 'community.contributors.range.all.caption',
        message: 'Repository history',
      }),
    },
  ]

  const [selectedRange, setSelectedRange] = useState<ContributorRankRange>('v03ToNow')
  const snapshot = contributorRankData[selectedRange]
  const topContributors = snapshot.entries.slice(0, 5)
  const selectedRangeLabel = rangeOptions.find(option => option.id === selectedRange)?.label ?? snapshot.label

  return (
    <Layout
      title={translate({
        id: 'community.contributors.pageTitle',
        message: 'Contributor Leaderboard',
      })}
      description={translate({
        id: 'community.contributors.pageDescription',
        message: 'vLLM Semantic Router contributor leaderboard by recent and historical repository commit activity.',
      })}
    >
      <main className={styles.container}>
        <header className={styles.header}>
          <div className={styles.titleBlock}>
            <p className={styles.eyebrow}>
              <Translate id="community.contributors.eyebrow">Community</Translate>
            </p>
            <div className={styles.titleRow}>
              <h1>
                <Translate id="community.contributors.h1">Contributor Leaderboard</Translate>
              </h1>
              <Link className={styles.contributeLink} to="/community/contributing">
                <Translate id="community.contributors.startContributing">Start contributing</Translate>
              </Link>
            </div>
          </div>
        </header>

        <section
          className={styles.metrics}
          aria-label={translate({ id: 'community.contributors.metrics.aria', message: 'Contributor rank summary' })}
        >
          <Metric
            label={translate({ id: 'community.contributors.metrics.contributors', message: 'Contributors' })}
            value={snapshot.totalContributors.toLocaleString(numberLocale)}
          />
          <Metric
            label={translate({ id: 'community.contributors.metrics.newContributors', message: 'New Contributors' })}
            value={snapshot.newContributors.toLocaleString(numberLocale)}
          />
          <Metric
            label={translate({ id: 'community.contributors.metrics.commits', message: 'Commits' })}
            value={snapshot.totalCommits.toLocaleString(numberLocale)}
          />
        </section>

        <section
          className={styles.podiumSection}
          aria-label={translate({ id: 'community.contributors.podium.aria', message: 'Contributor podium' })}
        >
          {topContributors.map(entry => (
            <TopContributorCard key={`${snapshot.id}-top-${entry.rank}`} entry={entry} numberLocale={numberLocale} />
          ))}
        </section>

        <section
          className={styles.leaderboardSection}
          aria-label={translate({
            id: 'community.contributors.leaderboard.aria',
            message: '{rangeLabel} contributor rank',
          }, { rangeLabel: selectedRangeLabel })}
        >
          <div className={styles.sectionHeader}>
            <div>
              <h2>{selectedRangeLabel}</h2>
              <p>
                {formatRange(snapshot.startDate, snapshot.endDate, dateLocale)}
                {' '}
                ·
                {' '}
                {translate({ id: 'community.contributors.updated', message: 'Updated' })}
                {' '}
                {formatDate(contributorRankGeneratedAt, dateLocale)}
              </p>
            </div>
            <label className={styles.rangeSelectLabel}>
              <span>
                <Translate id="community.contributors.range.label">Range</Translate>
              </span>
              <select
                className={styles.rangeSelect}
                value={selectedRange}
                aria-label={translate({
                  id: 'community.contributors.range.aria',
                  message: 'Contributor leaderboard release window',
                })}
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
            <span><Translate id="community.contributors.table.rank">Rank</Translate></span>
            <span><Translate id="community.contributors.table.contributor">Contributor</Translate></span>
            <span><Translate id="community.contributors.table.commits">Commits</Translate></span>
            <span><Translate id="community.contributors.table.share">Share</Translate></span>
            <span><Translate id="community.contributors.table.latest">Latest</Translate></span>
          </div>

          <div className={styles.rankList}>
            {snapshot.entries.map(entry => (
              <ContributorRow key={`${snapshot.id}-${entry.rank}-${entry.name}`} entry={entry} dateLocale={dateLocale} numberLocale={numberLocale} />
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

const TopContributorCard: React.FC<{ entry: ContributorRankEntry, numberLocale: string }> = ({ entry, numberLocale }) => {
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
              <span>
                <Translate id="community.contributors.gitAuthor">Git author</Translate>
              </span>
            )}
      </div>
      <div className={styles.podiumStats}>
        <strong>{entry.commits.toLocaleString(numberLocale)}</strong>
        <span>{formatPercent(entry.share)}</span>
      </div>
    </article>
  )
}

const ContributorRow: React.FC<{
  entry: ContributorRankEntry
  numberLocale: string
  dateLocale: string
}> = ({ entry, numberLocale, dateLocale }) => {
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
                <Translate id="community.contributors.newContributor">New Contributor</Translate>
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
                <span className={styles.handle}>
                  <Translate id="community.contributors.gitAuthor">Git author</Translate>
                </span>
              )}
        </div>
      </div>

      <div className={styles.statBlock}>
        <span><Translate id="community.contributors.table.commits">Commits</Translate></span>
        <strong>{entry.commits.toLocaleString(numberLocale)}</strong>
      </div>

      <div className={styles.share}>
        <span>{sharePercent}</span>
        <div className={styles.shareTrack} aria-hidden="true">
          <span className={styles.shareFill} style={{ width: barWidth }} />
        </div>
      </div>

      <div className={styles.statBlock}>
        <span><Translate id="community.contributors.table.latest">Latest</Translate></span>
        <strong>{formatDate(entry.latestCommitDate, dateLocale)}</strong>
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
      alt={translate({
        id: 'community.contributors.avatarAlt',
        message: '{name} avatar',
      }, { name: entry.name })}
      loading="lazy"
      onError={() => setDidFail(true)}
    />
  )
}

function formatRange(startDate: string | null, endDate: string, locale: string): string {
  if (!startDate) {
    return translate({
      id: 'community.contributors.range.through',
      message: 'Through {date}',
    }, { date: formatDate(endDate, locale) })
  }

  return `${formatDate(startDate, locale)} - ${formatDate(endDate, locale)}`
}

function formatDate(value: string, locale: string): string {
  return new Intl.DateTimeFormat(locale, {
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
