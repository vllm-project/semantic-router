import { execFileSync } from 'node:child_process'
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const scriptDir = dirname(fileURLToPath(import.meta.url))
const repoRoot = resolve(scriptDir, '..', '..')
const outputPath = resolve(repoRoot, 'website', 'src', 'data', 'committerReviewStats.generated.ts')
const teamMembersPath = resolve(repoRoot, 'website', 'src', 'data', 'teamMembers.tsx')
const githubOwner = 'vllm-project'
const githubRepo = 'semantic-router'
const releaseTags = {
  v01: 'v0.1.0',
  v02: 'v0.2.0',
  v03: 'v0.3.0',
}
const reviewStates = new Set(['APPROVED', 'CHANGES_REQUESTED', 'COMMENTED'])
const rangeIds = ['v03ToNow', 'v02ToV03', 'v01ToV02', 'v0ToV01', 'all']

const pullRequestsQuery = `
query($cursor: String) {
  repository(owner: "${githubOwner}", name: "${githubRepo}") {
    pullRequests(states: MERGED, first: 50, after: $cursor, orderBy: { field: UPDATED_AT, direction: DESC }) {
      pageInfo { hasNextPage endCursor }
      nodes {
        number
        mergedAt
        mergedBy { login }
        reviews(first: 100) {
          nodes {
            author { login }
            state
            submittedAt
          }
        }
      }
    }
  }
}`

try {
  const generatedAt = formatLocalDate(new Date())
  const committers = readCommitterDirectory()
  const committerLogins = new Set(committers.map(committer => normalizeLogin(committer.login)))
  const releaseTimeline = readReleaseTimeline()
  const rangeDefinitions = buildReleaseRangeDefinitions(releaseTimeline, generatedAt)
  const pullRequests = fetchMergedPullRequests()
  const snapshots = Object.fromEntries(
    rangeDefinitions.map(range => [
      range.id,
      buildSnapshot(range, generatedAt, committers, committerLogins, pullRequests),
    ]),
  )

  mkdirSync(dirname(outputPath), { recursive: true })
  writeFileSync(outputPath, renderTypeScript(generatedAt, snapshots))
}
catch (error) {
  if (existsSync(outputPath)) {
    console.warn(`Skipping committer review stats refresh: ${error.message}`)
    process.exit(0)
  }

  throw error
}

function readCommitterDirectory() {
  const source = readFileSync(teamMembersPath, 'utf8')
  const blocks = [
    sliceExportBlock(source, 'topNewContributorMembers'),
    sliceExportBlock(source, 'allCommitterMembers'),
  ]

  const byLogin = new Map()
  for (const block of blocks) {
    for (const entry of parseTeamMemberBlock(block)) {
      byLogin.set(normalizeLogin(entry.login), entry)
    }
  }

  return [...byLogin.values()].sort((left, right) => left.name.localeCompare(right.name))
}

function sliceExportBlock(source, exportName) {
  const start = source.indexOf(`export const ${exportName}`)
  if (start < 0) {
    return ''
  }

  const arrayStart = source.indexOf('= [', start)
  if (arrayStart < 0) {
    return ''
  }

  const bracketStart = arrayStart + 2
  let depth = 0

  for (let index = bracketStart; index < source.length; index += 1) {
    const char = source[index]
    if (char === '[') {
      depth += 1
    }
    if (char === ']') {
      depth -= 1
      if (depth === 0) {
        return source.slice(bracketStart, index + 1)
      }
    }
  }

  return ''
}

function parseTeamMemberBlock(block) {
  const entries = []
  const chunks = block.split('memberType: \'committer\'')

  for (const chunk of chunks.slice(0, -1)) {
    const nameMatch = chunk.match(/name: '((?:\\'|[^'])*)'/)
    const githubMatch = chunk.match(/github: 'https:\/\/github\.com\/([^']+)'/)

    if (!nameMatch || !githubMatch) {
      continue
    }

    const login = githubMatch[1]
    entries.push({
      name: nameMatch[1].replace(/\\'/g, '\''),
      login,
      avatarLogin: login,
      avatarUrl: `https://github.com/${login}.png?size=160`,
      avatarSeed: login,
    })
  }

  return entries
}

function fetchMergedPullRequests() {
  const pullRequests = []
  let cursor = null

  do {
    const page = graphqlRequest(pullRequestsQuery, { cursor })
    const connection = page.repository.pullRequests
    pullRequests.push(...connection.nodes)
    cursor = connection.pageInfo.hasNextPage ? connection.pageInfo.endCursor : null
  } while (cursor)

  return pullRequests
}

function graphqlRequest(query, variables) {
  const args = ['api', 'graphql', '-f', `query=${query}`]

  for (const [key, value] of Object.entries(variables)) {
    if (value != null) {
      args.push('-f', `${key}=${value}`)
    }
  }

  const output = execFileSync('gh', args, {
    cwd: repoRoot,
    encoding: 'utf8',
    maxBuffer: 1024 * 1024 * 32,
    stdio: ['ignore', 'pipe', 'pipe'],
  }).trim()

  const payload = JSON.parse(output)
  if (payload.errors?.length) {
    throw new Error(payload.errors.map(error => error.message).join('; '))
  }

  return payload.data
}

function buildSnapshot(range, generatedAt, committers, committerLogins, pullRequests) {
  const statsByLogin = new Map(
    committers.map(committer => [
      normalizeLogin(committer.login),
      {
        ...committer,
        login: committer.login,
        reviews: 0,
        merges: 0,
      },
    ]),
  )

  for (const pullRequest of pullRequests) {
    const mergedAt = pullRequest.mergedAt?.slice(0, 10) ?? null
    const mergedBy = normalizeLogin(pullRequest.mergedBy?.login)

    if (
      mergedAt
      && mergedBy
      && committerLogins.has(mergedBy)
      && !isBotLogin(mergedBy)
      && isDateInRange(mergedAt, range)
    ) {
      statsByLogin.get(mergedBy).merges += 1
    }

    const reviewedInPr = new Set()
    for (const review of pullRequest.reviews?.nodes ?? []) {
      const reviewer = normalizeLogin(review.author?.login)
      const submittedAt = review.submittedAt?.slice(0, 10) ?? null

      if (
        !reviewer
        || !committerLogins.has(reviewer)
        || isBotLogin(reviewer)
        || reviewedInPr.has(reviewer)
        || !reviewStates.has(review.state)
        || !submittedAt
        || !isDateInRange(submittedAt, range)
      ) {
        continue
      }

      reviewedInPr.add(reviewer)
      statsByLogin.get(reviewer).reviews += 1
    }
  }

  const entries = [...statsByLogin.values()]
    .map(entry => ({
      ...entry,
      total: entry.reviews + entry.merges,
    }))
    .filter(entry => entry.total > 0)
    .sort((left, right) => {
      if (right.total !== left.total) {
        return right.total - left.total
      }
      if (right.reviews !== left.reviews) {
        return right.reviews - left.reviews
      }
      return left.name.localeCompare(right.name)
    })
    .map((entry, index) => ({
      rank: index + 1,
      name: entry.name,
      login: entry.login,
      avatarLogin: entry.avatarLogin,
      avatarUrl: entry.avatarUrl,
      avatarSeed: entry.avatarSeed,
      key: `github:${normalizeLogin(entry.login)}`,
      reviews: entry.reviews,
      merges: entry.merges,
      total: entry.total,
    }))

  return {
    id: range.id,
    label: range.label,
    generatedAt,
    startDate: range.startDate,
    endDate: range.endDate,
    description: range.description,
    activeCommitters: entries.length,
    totalReviews: entries.reduce((sum, entry) => sum + entry.reviews, 0),
    totalMerges: entries.reduce((sum, entry) => sum + entry.merges, 0),
    entries,
  }
}

function buildReleaseRangeDefinitions(releaseTimeline, generatedAt) {
  if (!releaseTimeline) {
    return [{
      id: 'all',
      label: 'All time',
      startDate: null,
      endDate: generatedAt,
      description: 'Committer review and merge activity across repository history.',
    }]
  }

  return [
    {
      id: 'v03ToNow',
      label: 'v0.3 -> Now',
      startDate: releaseTimeline.v03.publishedAt.slice(0, 10),
      endDate: generatedAt,
      description: 'Committer review and merge activity after v0.3.0.',
    },
    {
      id: 'v02ToV03',
      label: 'v0.2 -> v0.3',
      startDate: releaseTimeline.v02.publishedAt.slice(0, 10),
      endDate: releaseTimeline.v03.publishedAt.slice(0, 10),
      description: 'Committer review and merge activity between v0.2.0 and v0.3.0.',
    },
    {
      id: 'v01ToV02',
      label: 'v0.1 -> v0.2',
      startDate: releaseTimeline.v01.publishedAt.slice(0, 10),
      endDate: releaseTimeline.v02.publishedAt.slice(0, 10),
      description: 'Committer review and merge activity between v0.1.0 and v0.2.0.',
    },
    {
      id: 'v0ToV01',
      label: 'v0 -> v0.1',
      startDate: null,
      endDate: releaseTimeline.v01.publishedAt.slice(0, 10),
      description: 'Initial committer review and merge activity through v0.1.0.',
    },
    {
      id: 'all',
      label: 'All time',
      startDate: null,
      endDate: generatedAt,
      description: 'Full committer review and merge history.',
    },
  ]
}

function readReleaseTimeline() {
  try {
    const output = execFileSync('gh', ['api', `repos/${githubOwner}/${githubRepo}/releases`], {
      cwd: repoRoot,
      encoding: 'utf8',
      maxBuffer: 1024 * 1024 * 4,
      stdio: ['ignore', 'pipe', 'pipe'],
    }).trim()
    const releases = JSON.parse(output).filter(release => !release.draft)
    const byTag = new Map(releases.map(release => [release.tag_name, release]))
    const v01 = byTag.get(releaseTags.v01)
    const v02 = byTag.get(releaseTags.v02)
    const v03 = byTag.get(releaseTags.v03)

    if (!v01 || !v02 || !v03) {
      return null
    }

    return {
      v01: { publishedAt: v01.published_at },
      v02: { publishedAt: v02.published_at },
      v03: { publishedAt: v03.published_at },
    }
  }
  catch {
    return null
  }
}

function isDateInRange(date, range) {
  return (!range.startDate || date >= range.startDate) && (!range.endDate || date <= range.endDate)
}

function isBotLogin(login) {
  return login.includes('[bot]') || login === 'dependabot' || login === 'github-actions'
}

function normalizeLogin(login) {
  return String(login ?? '').trim().toLowerCase()
}

function formatLocalDate(date) {
  const year = date.getFullYear()
  const month = String(date.getMonth() + 1).padStart(2, '0')
  const day = String(date.getDate()).padStart(2, '0')

  return `${year}-${month}-${day}`
}

function renderTypeScript(generatedAt, snapshots) {
  return `/* eslint-disable */
// This file is generated by \`npm run committers:stats\`.
// Source: GitHub GraphQL pull requests + reviews; committers from teamMembers.tsx.

export type CommitterStatsRange = ${rangeIds.map(id => `'${id}'`).join(' | ')}

export interface CommitterStatsEntry {
  rank: number
  key: string
  name: string
  login?: string
  avatarLogin?: string
  avatarUrl?: string
  avatarSeed: string
  reviews: number
  merges: number
  total: number
}

export interface CommitterStatsSnapshot {
  id: CommitterStatsRange
  label: string
  generatedAt: string
  startDate: string | null
  endDate: string
  description: string
  activeCommitters: number
  totalReviews: number
  totalMerges: number
  entries: CommitterStatsEntry[]
}

export const committerStatsGeneratedAt = '${generatedAt}'

export const committerStatsData = ${JSON.stringify(snapshots, null, 2)} satisfies Record<CommitterStatsRange, CommitterStatsSnapshot>
`
}
