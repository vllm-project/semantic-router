import { execFileSync } from 'node:child_process'
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const scriptDir = dirname(fileURLToPath(import.meta.url))
const repoRoot = resolve(scriptDir, '..', '..')
const outputPath = resolve(
  repoRoot,
  'website',
  'src',
  'data',
  'committerActivity.generated.ts',
)
const teamMembersPath = resolve(
  repoRoot,
  'website',
  'src',
  'data',
  'teamMembers.tsx',
)
const githubOwner = 'vllm-project'
const githubRepo = 'semantic-router'
const reviewStates = new Set(['APPROVED', 'CHANGES_REQUESTED', 'COMMENTED'])

const pullRequestsQuery = `
query($cursor: String) {
  repository(owner: "${githubOwner}", name: "${githubRepo}") {
    pullRequests(first: 50, after: $cursor, orderBy: { field: UPDATED_AT, direction: DESC }) {
      pageInfo { hasNextPage endCursor }
      nodes {
        number
        createdAt
        updatedAt
        author { login }
        reviews(last: 100) {
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

const issuesQuery = `
query($cursor: String) {
  repository(owner: "${githubOwner}", name: "${githubRepo}") {
    issues(first: 50, after: $cursor, orderBy: { field: UPDATED_AT, direction: DESC }) {
      pageInfo { hasNextPage endCursor }
      nodes {
        number
        createdAt
        updatedAt
        author { login }
        comments(last: 100) {
          nodes {
            author { login }
            createdAt
          }
        }
      }
    }
  }
}`

try {
  const generatedAt = formatDate(new Date())
  const cutoffDate = subtractCalendarMonths(generatedAt, 3)
  const committers = readCommitterDirectory()
  const activity = buildEmptyActivity(committers)

  recordPullRequestActivity(activity, cutoffDate)
  recordIssueActivity(activity, cutoffDate)

  const entries = committers.map((committer) => {
    const memberActivity = activity.get(normalizeLogin(committer.login))
    const pullRequests = memberActivity.pullRequests.size
    const reviews = memberActivity.reviews.size
    const issues = memberActivity.issues.size
    const total = pullRequests + reviews + issues

    return {
      name: committer.name,
      login: committer.login,
      pullRequests,
      reviews,
      issues,
      total,
      status: total > 0 ? 'active' : 'emeritus',
    }
  })

  mkdirSync(dirname(outputPath), { recursive: true })
  writeFileSync(outputPath, renderTypeScript(generatedAt, cutoffDate, entries))
}
catch (error) {
  if (existsSync(outputPath)) {
    console.warn(`Skipping committer activity refresh: ${error.message}`)
    process.exit(0)
  }

  throw error
}

function readCommitterDirectory() {
  const source = readFileSync(teamMembersPath, 'utf8')
  const exportNames = source.includes('export const allCommitterMembers')
    ? ['topNewContributorMembers', 'allCommitterMembers']
    : ['topNewContributorMembers', 'committerMembers']
  const byLogin = new Map()

  for (const exportName of exportNames) {
    const block = sliceExportBlock(source, exportName)
    for (const entry of parseTeamMemberBlock(block)) {
      byLogin.set(normalizeLogin(entry.login), entry)
    }
  }

  return [...byLogin.values()]
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

    entries.push({
      name: nameMatch[1].replace(/\\'/g, '\''),
      login: githubMatch[1],
    })
  }

  return entries
}

function buildEmptyActivity(committers) {
  return new Map(
    committers.map(committer => [
      normalizeLogin(committer.login),
      {
        pullRequests: new Set(),
        reviews: new Set(),
        issues: new Set(),
      },
    ]),
  )
}

function recordPullRequestActivity(activity, cutoffDate) {
  for (const pullRequest of fetchRecentNodes(
    pullRequestsQuery,
    'pullRequests',
    cutoffDate,
  )) {
    const author = normalizeLogin(pullRequest.author?.login)
    if (
      author
      && activity.has(author)
      && isOnOrAfter(pullRequest.createdAt, cutoffDate)
    ) {
      activity.get(author).pullRequests.add(pullRequest.number)
    }

    for (const review of pullRequest.reviews?.nodes ?? []) {
      const reviewer = normalizeLogin(review.author?.login)
      if (
        reviewer
        && activity.has(reviewer)
        && reviewStates.has(review.state)
        && isOnOrAfter(review.submittedAt, cutoffDate)
      ) {
        activity.get(reviewer).reviews.add(pullRequest.number)
      }
    }
  }
}

function recordIssueActivity(activity, cutoffDate) {
  for (const issue of fetchRecentNodes(issuesQuery, 'issues', cutoffDate)) {
    const author = normalizeLogin(issue.author?.login)
    if (author && activity.has(author) && isOnOrAfter(issue.createdAt, cutoffDate)) {
      activity.get(author).issues.add(issue.number)
    }

    for (const comment of issue.comments?.nodes ?? []) {
      const commenter = normalizeLogin(comment.author?.login)
      if (
        commenter
        && activity.has(commenter)
        && isOnOrAfter(comment.createdAt, cutoffDate)
      ) {
        activity.get(commenter).issues.add(issue.number)
      }
    }
  }
}

function fetchRecentNodes(query, connectionName, cutoffDate) {
  const nodes = []
  let cursor = null

  do {
    const page = graphqlRequest(query, { cursor })
    const connection = page.repository[connectionName]
    nodes.push(
      ...connection.nodes.filter(node => isOnOrAfter(node.updatedAt, cutoffDate)),
    )

    const reachedCutoff = connection.nodes.some(
      node => !isOnOrAfter(node.updatedAt, cutoffDate),
    )
    cursor
      = connection.pageInfo.hasNextPage && !reachedCutoff
        ? connection.pageInfo.endCursor
        : null
  } while (cursor)

  return nodes
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

function subtractCalendarMonths(date, months) {
  const value = new Date(`${date}T00:00:00Z`)
  value.setUTCMonth(value.getUTCMonth() - months)
  return formatDate(value)
}

function isOnOrAfter(timestamp, cutoffDate) {
  return Boolean(timestamp && timestamp.slice(0, 10) >= cutoffDate)
}

function normalizeLogin(login) {
  return login?.trim().toLowerCase() ?? ''
}

function formatDate(date) {
  return date.toISOString().slice(0, 10)
}

function renderTypeScript(generatedAt, cutoffDate, entries) {
  const activeCount = entries.filter(entry => entry.status === 'active').length
  const emeritusCount = entries.length - activeCount

  return `/* eslint-disable */
// This file is generated by \`npm run committers:activity\`.
// Activity includes authored PRs, submitted reviews, and authored or commented issues.

export type CommitterActivityStatus = 'active' | 'emeritus'

export interface CommitterActivityEntry {
  name: string
  login: string
  pullRequests: number
  reviews: number
  issues: number
  total: number
  status: CommitterActivityStatus
}

export const committerActivityWindow = ${JSON.stringify({
  generatedAt,
  cutoffDate,
  months: 3,
  activeCount,
  emeritusCount,
}, null, 2)}

export const committerActivityEntries: CommitterActivityEntry[] = ${JSON.stringify(entries, null, 2)}
`
}
