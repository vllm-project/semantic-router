import React, { useMemo, useState } from 'react'
import Layout from '@theme/Layout'
import Translate, { translate } from '@docusaurus/Translate'
import useDocusaurusContext from '@docusaurus/useDocusaurusContext'
import { FaGithub } from 'react-icons/fa'
import CommunityLayout from '@site/src/components/community/CommunityLayout'
import {
  contributorRankData,
  contributorRankGeneratedAt,
} from '../../data/contributorRank.generated'
import type {
  ContributorRankEntry,
  ContributorRankRange,
} from '../../data/contributorRank.generated'
import styles from './contributors.module.css'

type SortBy = 'commits' | 'reviews'

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
        message: 'v0.3 → Now',
      }),
      caption: translate({
        id: 'community.contributors.range.v03ToNow.caption',
        message: 'Since v0.3',
      }),
    },
    {
      id: 'v02ToV03',
      label: translate({
        id: 'community.contributors.range.v02ToV03.label',
        message: 'v0.2 → v0.3',
      }),
      caption: translate({
        id: 'community.contributors.range.v02ToV03.caption',
        message: 'Between releases',
      }),
    },
    {
      id: 'v01ToV02',
      label: translate({
        id: 'community.contributors.range.v01ToV02.label',
        message: 'v0.1 → v0.2',
      }),
      caption: translate({
        id: 'community.contributors.range.v01ToV02.caption',
        message: 'Between releases',
      }),
    },
    {
      id: 'v0ToV01',
      label: translate({
        id: 'community.contributors.range.v0ToV01.label',
        message: 'v0 → v0.1',
      }),
      caption: translate({
        id: 'community.contributors.range.v0ToV01.caption',
        message: 'Project start',
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
  const [sortBy, setSortBy] = useState<SortBy>('commits')
  const snapshot = contributorRankData[selectedRange]
  const selectedRangeLabel = rangeOptions.find(option => option.id === selectedRange)?.label ?? snapshot.label
  const commitAuthors = snapshot.entries.filter(entry => entry.commits > 0).length

  const rankedEntries = useMemo(() => {
    const sorted = [...snapshot.entries].sort((left, right) => {
      if (sortBy === 'reviews') {
        if (right.reviews !== left.reviews) {
          return right.reviews - left.reviews
        }

        if (right.commits !== left.commits) {
          return right.commits - left.commits
        }

        return left.rank - right.rank
      }

      if (right.commits !== left.commits) {
        return right.commits - left.commits
      }

      if (right.reviews !== left.reviews) {
        return right.reviews - left.reviews
      }

      return left.rank - right.rank
    })

    return sorted.map((entry, index) => ({
      ...entry,
      rank: index + 1,
    }))
  }, [snapshot.entries, sortBy])

  const topContributors = rankedEntries.slice(0, 5)

  return (
    <Layout
      title={translate({
        id: 'community.contributors.pageTitle',
        message: 'Contributor Leaderboard',
      })}
      description={translate({
        id: 'community.contributors.pageDescription',
        message: 'Explore vLLM Semantic Router contributors by commit and pull request review activity across release windows.',
      })}
    >
      <CommunityLayout
        activeKey="leaderboard"
        title={<Translate id="community.contributors.h1">Contributor Leaderboard</Translate>}
      >
        <section
          className={styles.metrics}
          aria-label={translate({ id: 'community.contributors.metrics.aria', message: 'Contributor rank summary' })}
        >
          <Metric
            label={translate({ id: 'community.contributors.metrics.contributors', message: 'Contributors' })}
            value={snapshot.totalContributors.toLocaleString(numberLocale)}
          />
          <Metric
            label={selectedRange === 'all'
              ? translate({ id: 'community.contributors.metrics.commitAuthors', message: 'Commit authors' })
              : translate({ id: 'community.contributors.metrics.newContributors', message: 'First-time commit authors' })}
            value={(selectedRange === 'all' ? commitAuthors : snapshot.newContributors).toLocaleString(numberLocale)}
          />
          <Metric
            label={translate({ id: 'community.contributors.metrics.commits', message: 'Commits' })}
            value={snapshot.totalCommits.toLocaleString(numberLocale)}
          />
          <Metric
            label={translate({ id: 'community.contributors.metrics.reviews', message: 'Reviews' })}
            value={snapshot.totalReviews.toLocaleString(numberLocale)}
          />
        </section>

        <section
          className={styles.podiumSection}
          aria-label={translate({ id: 'community.contributors.podium.aria', message: 'Contributor podium' })}
        >
          {topContributors.map(entry => (
            <TopContributorCard
              key={`${snapshot.id}-top-${entry.rank}`}
              entry={entry}
              numberLocale={numberLocale}
              sortBy={sortBy}
            />
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
              <p className={styles.methodNote}>
                <Translate id="community.contributors.method">
                  Contributors include qualifying commit authors and reviewers. “New” means a first repository commit; reviews count once per contributor per merged pull request.
                </Translate>
              </p>
            </div>
            <div className={styles.sectionControls}>
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
              <label className={styles.rangeSelectLabel}>
                <span>
                  <Translate id="community.contributors.sort.label">Sort by</Translate>
                </span>
                <select
                  className={styles.rangeSelect}
                  value={sortBy}
                  aria-label={translate({
                    id: 'community.contributors.sort.aria',
                    message: 'Contributor leaderboard sort order',
                  })}
                  onChange={event => setSortBy(event.target.value as SortBy)}
                >
                  <option value="commits">
                    {translate({
                      id: 'community.contributors.sort.commits',
                      message: 'Commits',
                    })}
                  </option>
                  <option value="reviews">
                    {translate({
                      id: 'community.contributors.sort.reviews',
                      message: 'Reviews',
                    })}
                  </option>
                </select>
              </label>
            </div>
          </div>

          <div className={styles.rankListHeader} aria-hidden="true">
            <span><Translate id="community.contributors.table.rank">Rank</Translate></span>
            <span><Translate id="community.contributors.table.contributor">Contributor</Translate></span>
            <span><Translate id="community.contributors.table.commits">Commits</Translate></span>
            <span><Translate id="community.contributors.table.reviews">Reviews</Translate></span>
            <span>
              {sortBy === 'reviews'
                ? <Translate id="community.contributors.table.reviewShare">Review share</Translate>
                : <Translate id="community.contributors.table.commitShare">Commit share</Translate>}
            </span>
            <span><Translate id="community.contributors.table.latest">Latest commit</Translate></span>
          </div>

          <div className={styles.rankList}>
            {rankedEntries.map(entry => (
              <ContributorRow
                key={`${snapshot.id}-${entry.rank}-${entry.name}`}
                entry={entry}
                dateLocale={dateLocale}
                numberLocale={numberLocale}
                showNewContributorStatus={selectedRange !== 'all'}
                activityShare={sortBy === 'reviews'
                  ? (snapshot.totalReviews > 0 ? entry.reviews / snapshot.totalReviews : 0)
                  : entry.share}
              />
            ))}
          </div>
        </section>
      </CommunityLayout>
    </Layout>
  )
}

const Metric: React.FC<{ label: string, value: string }> = ({ label, value }) => (
  <div className={styles.metric}>
    <span>{label}</span>
    <strong>{value}</strong>
  </div>
)

const TopContributorCard: React.FC<{
  entry: ContributorRankEntry
  numberLocale: string
  sortBy: SortBy
}> = ({ entry, numberLocale, sortBy }) => {
  const profileUrl = entry.login ? `https://github.com/${entry.login}` : undefined
  const activityCount = sortBy === 'reviews' ? entry.reviews : entry.commits

  return (
    <article className={styles.podiumCard}>
      <span className={styles.podiumRank}>{formatRankNumber(entry.rank)}</span>
      <ContributorAvatar entry={entry} />
      <div className={styles.podiumIdentity}>
        <span className={styles.podiumName}>{entry.name}</span>
        {profileUrl && entry.login && (
          <a href={profileUrl} target="_blank" rel="noopener noreferrer">
            <FaGithub aria-hidden="true" />
            {entry.login}
          </a>
        )}
      </div>
      <span className={styles.podiumActivity}>
        <strong>{activityCount.toLocaleString(numberLocale)}</strong>
        <span>
          {sortBy === 'reviews'
            ? <Translate id="community.contributors.sort.reviews">Reviews</Translate>
            : <Translate id="community.contributors.sort.commits">Commits</Translate>}
        </span>
      </span>
    </article>
  )
}

const ContributorRow: React.FC<{
  entry: ContributorRankEntry
  numberLocale: string
  dateLocale: string
  showNewContributorStatus: boolean
  activityShare: number
}> = ({ entry, numberLocale, dateLocale, showNewContributorStatus, activityShare }) => {
  const profileUrl = entry.login ? `https://github.com/${entry.login}` : undefined
  const sharePercent = formatPercent(activityShare)
  const barWidth = activityShare > 0 ? `${Math.max(activityShare * 100, 1.5)}%` : '0%'
  const isNewContributor = showNewContributorStatus && entry.isNewContributorSinceRelease

  return (
    <article className={`${styles.rankItem} ${isNewContributor ? styles.rankItemNew : ''}`}>
      <span className={styles.rankBadge}>
        {formatRankNumber(entry.rank)}
      </span>

      <div className={styles.contributor}>
        <ContributorAvatar entry={entry} />
        <div className={styles.identity}>
          <span className={styles.nameLine}>
            <span className={styles.name}>{entry.name}</span>
            {isNewContributor && (
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

      <div className={styles.statBlock}>
        <span><Translate id="community.contributors.table.reviews">Reviews</Translate></span>
        <strong>{entry.reviews.toLocaleString(numberLocale)}</strong>
      </div>

      <div className={styles.share}>
        <span>{sharePercent}</span>
        <div className={styles.shareTrack} aria-hidden="true">
          <span className={styles.shareFill} style={{ width: barWidth }} />
        </div>
      </div>

      <div className={styles.statBlock}>
        <span><Translate id="community.contributors.table.latest">Latest commit</Translate></span>
        <strong>{entry.commits > 0 ? formatDate(entry.latestCommitDate, dateLocale) : '—'}</strong>
      </div>
    </article>
  )
}

const ContributorAvatar: React.FC<{
  entry: ContributorRankEntry
}> = ({ entry }) => {
  const [didFail, setDidFail] = useState(false)
  const fallbackUrl = createFallbackAvatar(entry.avatarSeed || entry.name)
  const githubAvatarUrl = entry.avatarUrl ?? (entry.avatarLogin ? `https://github.com/${entry.avatarLogin}.png?size=160` : undefined)
  const avatarUrl = githubAvatarUrl && !didFail ? githubAvatarUrl : fallbackUrl

  return (
    <img
      className={styles.avatar}
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
