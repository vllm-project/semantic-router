import { execFileSync } from 'node:child_process'
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const scriptDir = dirname(fileURLToPath(import.meta.url))
const repoRoot = resolve(scriptDir, '..', '..')
const outputPath = resolve(repoRoot, 'website', 'src', 'data', 'contributorRank.generated.ts')
const githubRepo = 'vllm-project/semantic-router'
const releaseTags = {
  v01: 'v0.1.0',
  v02: 'v0.2.0',
  v03: 'v0.3.0',
}
const reviewStates = new Set(['APPROVED', 'CHANGES_REQUESTED', 'COMMENTED'])
const GH_MAX_ATTEMPTS = Number(process.env.GH_API_MAX_ATTEMPTS || 6)
const GH_RETRY_BASE_MS = Number(process.env.GH_API_RETRY_BASE_MS || 2000)
const GH_PAGE_DELAY_MS = Number(process.env.GH_API_PAGE_DELAY_MS || 400)

function sleepSync(ms) {
  if (!Number.isFinite(ms) || ms <= 0) {
    return
  }

  Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, Math.trunc(ms))
}

function errorText(error) {
  return [error?.message, error?.stderr, error?.stdout]
    .filter(Boolean)
    .map(value => String(value))
    .join('\n')
}

function isRetryableGhError(error) {
  return /(?:\b429\b|\b502\b|\b503\b|\b504\b|rate[- ]?limit|Bad Gateway|Gateway Time-out|secondary rate limit|ECONNRESET|ETIMEDOUT|EAI_AGAIN|socket hang up|HTTP 5\d\d)/i
    .test(errorText(error))
}

function runGh(args, options = {}) {
  const {
    maxBuffer = 1024 * 1024 * 8,
    attempts = GH_MAX_ATTEMPTS,
  } = options

  let lastError

  for (let attempt = 1; attempt <= attempts; attempt += 1) {
    try {
      return execFileSync('gh', args, {
        cwd: repoRoot,
        encoding: 'utf8',
        maxBuffer,
        stdio: ['ignore', 'pipe', 'pipe'],
      }).trim()
    }
    catch (error) {
      lastError = error
      if (attempt >= attempts || !isRetryableGhError(error)) {
        throw error
      }

      const delay = GH_RETRY_BASE_MS * (2 ** (attempt - 1))
      const summary = errorText(error).split('\n').find(Boolean) || 'unknown error'
      console.warn(
        `gh ${args.slice(0, 3).join(' ')} failed (attempt ${attempt}/${attempts}): ${summary}; retrying in ${delay}ms`,
      )
      sleepSync(delay)
    }
  }

  throw lastError
}

function ensureGitHubAuth() {
  if (!(process.env.GH_TOKEN || process.env.GITHUB_TOKEN)) {
    console.warn('GH_TOKEN/GITHUB_TOKEN unset; unauthenticated GitHub API limits are very low')
  }

  try {
    const output = runGh(['api', 'rate_limit'], { maxBuffer: 1024 * 1024, attempts: 3 })
    const core = JSON.parse(output)?.resources?.core
    if (!core) {
      return
    }

    console.log(`GitHub API rate limit: ${core.remaining}/${core.limit} remaining`)
    if (core.remaining < 50) {
      console.warn('GitHub core rate limit is low; requests may still fail after retries')
    }
  }
  catch (error) {
    console.warn(`Unable to check GitHub rate limit: ${errorText(error).split('\n')[0]}`)
  }
}

const pullRequestsQuery = `
query($cursor: String) {
  repository(owner: "vllm-project", name: "semantic-router") {
    pullRequests(states: MERGED, first: 50, after: $cursor, orderBy: { field: UPDATED_AT, direction: DESC }) {
      pageInfo { hasNextPage endCursor }
      nodes {
        number
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

const identityOverrides = [
  {
    key: 'xunzhuo-liu',
    name: 'Xunzhuo Liu',
    login: 'Xunzhuo',
    emails: ['bitliu@tencent.com', 'xunzhuo@vllm-semantic-router.ai', 'mixdeers@gmail.com'],
    names: ['bitliu', 'Xunzhuo'],
  },
  {
    key: 'huamin-chen',
    name: 'Huamin Chen',
    login: 'rootfs',
    emails: ['rootfs@users.noreply.github.com', 'hchen@redhat.com', 'huaminchen@microsoft.com'],
  },
  {
    key: 'samzong',
    name: 'samzong',
    login: 'samzong',
    emails: ['samzong.lu@gmail.com'],
  },
  {
    key: 'yuluo-yx',
    name: 'yuluo-yx',
    login: 'yuluo-yx',
    emails: ['yuluo08290126@gmail.com'],
    names: ['shown'],
  },
  {
    key: 'cryo-zd',
    name: 'cryo-zd',
    login: 'cryo-zd',
    emails: ['zdtna412@gmail.com'],
    names: ['cryo'],
  },
  {
    key: 'jaredforreal',
    name: 'JaredforReal',
    login: 'JaredforReal',
    emails: ['w13431838023@gmail.com'],
    names: ['Jared'],
  },
  {
    key: 'jintao-zhang',
    name: 'Jintao Zhang',
    login: 'tao12345666333',
    emails: ['zhangjintao9020@gmail.com'],
  },
  {
    key: 'senan-zedan',
    name: 'Senan Zedan',
    login: 'szedan-rh',
    emails: ['szedan@redhat.com'],
  },
  {
    key: 'liav-weiss',
    name: 'Liav Weiss',
    login: 'liavweiss',
    emails: ['74174727+liavweiss@users.noreply.github.com'],
  },
  {
    key: 'yehudit-kerido',
    name: 'Yehudit',
    login: 'yehuditkerido',
    emails: [
      '34643974+yehudit1987@users.noreply.github.com',
      '34643974+yehuditkerido@users.noreply.github.com',
    ],
    names: ['yehudit1987'],
  },
  {
    key: 'noa-limoy',
    name: 'Noa Limoy',
    login: 'noalimoy',
    emails: ['84776878+noalimoy@users.noreply.github.com', 'nlimoy@redhat.com'],
    names: ['noalimoy'],
  },
  {
    key: 'asaadbalum',
    name: 'asaadbalum',
    login: 'asaadbalum',
    emails: [
      '154635253+asaadbalum@users.noreply.github.com',
      'asaad.balum@gmail.com',
    ],
  },
  {
    key: 'abdallahsamabd',
    name: 'abdallahsamabd',
    login: 'abdallahsamabd',
    emails: ['42250800+abdallahsamabd@users.noreply.github.com'],
  },
  {
    key: 'onezero-y',
    name: 'OneZero-Y',
    login: 'OneZero-Y',
    emails: ['aukovyps@163.com'],
  },
  {
    key: 'haowu1234',
    name: 'haowu1234',
    login: 'haowu1234',
    emails: ['126473953+haowu1234@users.noreply.github.com'],
  },
  {
    key: 'faust',
    name: 'FAUST',
    login: 'FAUST-BENCHOU',
    emails: ['2319109590@qq.com'],
    names: ['FAUST.', 'zhoujinyu'],
  },
  {
    key: 'yue-zhu',
    name: 'Yue Zhu',
    login: 'yuezhu1',
    emails: ['16687552+yuezhu1@users.noreply.github.com'],
  },
  {
    key: 'henschwartz',
    name: 'henschwartz',
    login: 'henschwartz',
    emails: ['hschwart@redhat.com'],
  },
  {
    key: 'yangw',
    name: 'yangw',
    login: 'drivebyer',
    emails: ['wuyangmuc@gmail.com', 'yang.wu@daocloud.io'],
    names: ['drivebyer'],
  },
  {
    key: 'aayush-saini',
    name: 'Aayush Saini',
    login: 'AayushSaini101',
    emails: [
      '60972989+AayushSaini101@users.noreply.github.com',
      'kumaraayush9810@gmail.com',
    ],
    names: ['Moderator'],
  },
  {
    key: 'qiping-pan',
    name: 'Qiping Pan',
    login: 'ppppqp',
    emails: ['60682078+ppppqp@users.noreply.github.com'],
  },
  {
    key: 'marina-koushnir',
    name: 'Marina Koushnir',
    login: 'mkoushni',
    emails: ['mkoushni@redhat.com'],
  },
  {
    key: 'brent-salisbury',
    name: 'Brent Salisbury',
    login: 'bsalisbu',
    emails: ['bsalisbu@redhat.com', 'brent.salisbury@gmail.com'],
  },
  {
    key: 'yossi-ovadia',
    name: 'Yossi Ovadia',
    login: 'yossiovadia',
    emails: ['yovadia@redhat.com', 'yossi.ovadia@nokia.com'],
    names: ['yovadia'],
  },
  {
    key: 'srinivas-a',
    name: 'Srinivas A',
    login: 'srini-abhiram',
    emails: ['56465971+srini-abhiram@users.noreply.github.com'],
  },
  {
    key: 'chen-wang',
    name: 'Chen Wang',
    login: 'wangchen615',
    emails: ['Chen.Wang1@ibm.com'],
  },
  {
    key: 'ramakrishnan-sathyavageeswaran',
    name: 'Ramakrishnan Sathyavageeswaran',
    login: 'ramkrishs',
    emails: ['ramkrishs@outlook.com'],
  },
  {
    key: 'david-shrader',
    name: 'David Shrader',
    login: 'shraderdm',
    emails: ['shraderdm@gmail.com'],
  },
  {
    key: 'liuqihao',
    name: '刘启灏',
    login: 'BruceLoveDecimal',
    emails: ['liuqihao@liuqihaodemacbook-pro.local'],
  },
  {
    key: 'njx',
    name: 'NJX',
    emails: ['3771829673@qq.com'],
  },
  {
    key: 'rutuja-pathade',
    name: 'Rutuja Pathade',
    login: 'rpathade',
    emails: ['73137503+rpathade@users.noreply.github.com'],
  },
  {
    key: 'kaveesh-khattar',
    name: 'Kaveesh Khattar',
    emails: ['kaveeshkhattar@gmail.com'],
  },
]

const overrideByEmail = new Map()
const overrideByName = new Map()
const overrideByLogin = new Map()
const githubProfileByLogin = new Map()
const pullAuthorBySha = new Map()

for (const identity of identityOverrides) {
  if (identity.login) {
    overrideByLogin.set(normalizeLogin(identity.login), identity)
  }

  if (identity.avatarLogin) {
    overrideByLogin.set(normalizeLogin(identity.avatarLogin), identity)
  }

  for (const email of identity.emails ?? []) {
    overrideByEmail.set(normalizeEmail(email), identity)
  }

  for (const name of identity.names ?? []) {
    overrideByName.set(normalizeName(name), identity)
  }

  overrideByName.set(normalizeName(identity.name), identity)
}

try {
  ensureGitHubAuth()
  const generatedAt = formatLocalDate(new Date())
  const allRows = readContributorRows(null)
  const releaseTimeline = readReleaseTimeline()

  if (!releaseTimeline) {
    throw new Error('Release metadata is unavailable for contributor rank release windows.')
  }

  const releaseWindow = {
    baseRelease: releaseTimeline.v02,
    targetRelease: releaseTimeline.v03,
  }
  const rangeDefinitions = buildReleaseRangeDefinitions(releaseTimeline, generatedAt)
  let pullRequests = []

  try {
    pullRequests = fetchMergedPullRequests()
  }
  catch (error) {
    console.warn(`Skipping PR review stats: ${error.message}`)
  }

  const newContributorsSinceRelease = buildNewContributorsSinceRelease(releaseWindow, allRows, pullRequests)
  const snapshots = Object.fromEntries(
    rangeDefinitions.map(range => [range.id, buildSnapshot(range, generatedAt, allRows, pullRequests)]),
  )

  mkdirSync(dirname(outputPath), { recursive: true })
  writeFileSync(outputPath, renderTypeScript(generatedAt, snapshots, newContributorsSinceRelease))
}
catch (error) {
  if (existsSync(outputPath)) {
    console.warn(`Skipping contributor rank refresh: ${error.message}`)
    process.exit(0)
  }

  throw error
}

function buildReleaseRangeDefinitions(releaseTimeline, generatedAt) {
  return [
    {
      id: 'v03ToNow',
      label: 'v0.3 -> Now',
      startTagName: releaseTimeline.v03.tagName,
      endTagName: null,
      startDate: releaseTimeline.v03.publishedAt.slice(0, 10),
      endDate: generatedAt,
      description: 'Current non-merge commit activity after v0.3.0.',
    },
    {
      id: 'v02ToV03',
      label: 'v0.2 -> v0.3',
      startTagName: releaseTimeline.v02.tagName,
      endTagName: releaseTimeline.v03.tagName,
      startDate: releaseTimeline.v02.publishedAt.slice(0, 10),
      endDate: releaseTimeline.v03.publishedAt.slice(0, 10),
      description: 'Non-merge commit activity between v0.2.0 and v0.3.0.',
    },
    {
      id: 'v01ToV02',
      label: 'v0.1 -> v0.2',
      startTagName: releaseTimeline.v01.tagName,
      endTagName: releaseTimeline.v02.tagName,
      startDate: releaseTimeline.v01.publishedAt.slice(0, 10),
      endDate: releaseTimeline.v02.publishedAt.slice(0, 10),
      description: 'Non-merge commit activity between v0.1.0 and v0.2.0.',
    },
    {
      id: 'v0ToV01',
      label: 'v0 -> v0.1',
      startTagName: null,
      endTagName: releaseTimeline.v01.tagName,
      startDate: null,
      endDate: releaseTimeline.v01.publishedAt.slice(0, 10),
      description: 'Initial non-merge commit activity through v0.1.0.',
    },
    {
      id: 'all',
      label: 'All time',
      startTagName: null,
      endTagName: null,
      startDate: null,
      endDate: generatedAt,
      description: 'Full repository non-merge commit history.',
    },
  ]
}

function buildSnapshot(range, generatedAt, allRows, pullRequests) {
  const rows = readRowsForRange(range, allRows)
  const byContributor = collectContributorStats(rows)
  const newContributorKeys = readNewContributorKeysForRange(range, allRows, byContributor)
  const totalCommits = [...byContributor.values()].reduce((sum, entry) => sum + entry.commits, 0)
  const reviewStats = collectReviewStatsByContributorKey(pullRequests, range)
  const entries = rankContributorEntries(byContributor, totalCommits, newContributorKeys, reviewStats)
  const newContributors = entries.filter(entry => entry.isNewContributorSinceRelease).length
  const totalReviews = entries.reduce((sum, entry) => sum + entry.reviews, 0)

  return {
    id: range.id,
    label: range.label,
    generatedAt,
    startDate: range.startDate,
    endDate: range.endDate,
    description: range.description,
    totalCommits,
    totalReviews,
    totalContributors: entries.length,
    newContributors,
    entries,
  }
}

function readRowsForRange(range, allRows) {
  if (!range.startTagName && !range.endTagName) {
    return allRows
  }

  const commitShas = readCommitShasForRange(range)
  if (commitShas) {
    return allRows.filter(row => row.sha && commitShas.has(row.sha))
  }

  return filterRowsByDateRange(allRows, range.startDate, range.endDate)
}

function readCommitShasForRange(range) {
  if (range.startTagName && range.endTagName) {
    return readCommitShasBetweenTags(range.startTagName, range.endTagName)
  }

  if (range.startTagName) {
    return readCommitShasSinceTag(range.startTagName)
  }

  if (range.endTagName) {
    return readCommitShasThroughTag(range.endTagName)
  }

  return null
}

function readNewContributorKeysForRange(range, allRows, byContributor) {
  if (!range.startTagName && !range.startDate) {
    return new Set(byContributor.keys())
  }

  const historicalRows = readRowsBeforeRange(range, allRows)
  const historicalContributorKeys = new Set(collectContributorStats(historicalRows).keys())

  return new Set([...byContributor.keys()].filter(key => !historicalContributorKeys.has(key)))
}

function readRowsBeforeRange(range, allRows) {
  if (range.startTagName) {
    const commitShas = readCommitShasThroughTag(range.startTagName)
    if (commitShas) {
      return allRows.filter(row => row.sha && commitShas.has(row.sha))
    }
  }

  if (!range.startDate) {
    return []
  }

  return allRows.filter(row => row.date.slice(0, 10) < range.startDate)
}

function buildNewContributorsSinceRelease(releaseWindow, allRows, pullRequests = []) {
  if (!releaseWindow) {
    return {
      tagName: null,
      releaseName: null,
      releaseDate: null,
      targetTagName: null,
      targetReleaseName: null,
      targetReleaseDate: null,
      comparisonMode: 'none',
      totalCommits: 0,
      totalContributors: 0,
      entries: [],
    }
  }

  const { baseRelease, targetRelease } = releaseWindow
  const commitShasInWindow = readCommitShasBetweenTags(baseRelease.tagName, targetRelease.tagName)
  const rowsInWindow = commitShasInWindow
    ? allRows.filter(row => row.sha && commitShasInWindow.has(row.sha))
    : allRows.filter(row => row.date >= baseRelease.publishedAt && row.date <= targetRelease.publishedAt)
  const rowsBeforeBaseRelease = allRows.filter(row => row.date < baseRelease.publishedAt)
  const historicalContributorKeys = new Set(collectContributorStats(rowsBeforeBaseRelease).keys())
  const byContributor = collectContributorStats(rowsInWindow)

  for (const key of historicalContributorKeys) {
    byContributor.delete(key)
  }

  const totalCommits = [...byContributor.values()].reduce((sum, entry) => sum + entry.commits, 0)
  const reviewStats = collectReviewStatsByContributorKey(pullRequests, {
    startDate: baseRelease.publishedAt.slice(0, 10),
    endDate: targetRelease.publishedAt.slice(0, 10),
  })
  const entries = rankContributorEntries(byContributor, totalCommits, new Set(byContributor.keys()), reviewStats)

  return {
    tagName: baseRelease.tagName,
    releaseName: baseRelease.name,
    releaseDate: baseRelease.publishedAt.slice(0, 10),
    targetTagName: targetRelease.tagName,
    targetReleaseName: targetRelease.name,
    targetReleaseDate: targetRelease.publishedAt.slice(0, 10),
    comparisonMode: commitShasInWindow ? 'tag' : 'date',
    totalCommits,
    totalContributors: entries.length,
    entries,
  }
}

function readExistingNewContributorsSinceRelease() {
  if (!existsSync(outputPath)) {
    return null
  }

  try {
    const source = readFileSync(outputPath, 'utf8')
    const match = source.match(/export const newContributorsSinceRelease = ([\s\S]*?) satisfies NewContributorsSinceReleaseSnapshot/)
    if (!match?.[1]) {
      return null
    }

    const snapshot = JSON.parse(match[1])
    if (!Array.isArray(snapshot.entries) || snapshot.entries.length === 0) {
      return null
    }

    console.warn('Reusing existing newContributorsSinceRelease snapshot because release window metadata is unavailable.')

    return snapshot
  }
  catch (error) {
    console.warn(`Unable to reuse existing newContributorsSinceRelease snapshot: ${error.message}`)

    return null
  }
}

function collectContributorStats(rows) {
  const byContributor = new Map()

  for (const row of rows) {
    const identity = resolveIdentity(row.name, row.email, row)

    if (identity.isBot) {
      continue
    }

    const current = byContributor.get(identity.key) ?? {
      key: identity.key,
      name: identity.name,
      login: identity.login,
      avatarLogin: identity.avatarLogin,
      avatarUrl: identity.avatarUrl,
      avatarSeed: identity.avatarSeed,
      commits: 0,
      firstCommitDate: row.date,
      latestCommitDate: row.date,
    }

    current.commits += 1
    current.firstCommitDate = minDate(current.firstCommitDate, row.date)
    current.latestCommitDate = maxDate(current.latestCommitDate, row.date)
    byContributor.set(identity.key, current)
  }

  return byContributor
}

function rankContributorEntries(byContributor, totalCommits, newContributorKeys, reviewStats = new Map()) {
  const merged = mergeReviewOnlyContributors(byContributor, reviewStats)

  return [...merged.values()]
    .map(entry => ({
      rank: 0,
      name: entry.name,
      login: entry.login,
      avatarLogin: entry.avatarLogin,
      avatarUrl: entry.avatarUrl,
      avatarSeed: entry.avatarSeed,
      key: entry.key,
      commits: entry.commits,
      reviews: reviewStats.get(entry.key)?.count ?? 0,
      share: totalCommits > 0 ? Number((entry.commits / totalCommits).toFixed(4)) : 0,
      firstCommitDate: entry.firstCommitDate.slice(0, 10),
      latestCommitDate: entry.latestCommitDate.slice(0, 10),
      isNewContributorSinceRelease: newContributorKeys.has(entry.key),
    }))
    .filter(entry => entry.commits > 0 || entry.reviews > 0)
    .sort((left, right) => {
      if (right.commits !== left.commits) {
        return right.commits - left.commits
      }

      if (right.reviews !== left.reviews) {
        return right.reviews - left.reviews
      }

      return left.name.localeCompare(right.name)
    })
    .map((entry, index) => ({
      ...entry,
      rank: index + 1,
    }))
}

function mergeReviewOnlyContributors(byContributor, reviewStats) {
  const merged = new Map(byContributor)

  for (const [key, review] of reviewStats) {
    if (merged.has(key)) {
      continue
    }

    const login = review.login
    const override = overrideByLogin.get(normalizeLogin(login))
    const profile = login ? readGitHubUserProfile(login) : null
    const resolvedLogin = profile?.login ?? login

    merged.set(key, {
      key,
      name: override?.name ?? profile?.name ?? resolvedLogin ?? 'Unknown contributor',
      login: resolvedLogin,
      avatarLogin: override?.avatarLogin ?? resolvedLogin,
      avatarUrl: profile?.avatarUrl,
      avatarSeed: override?.key ?? normalizeLogin(resolvedLogin ?? key),
      commits: 0,
      firstCommitDate: `${review.latestDate}T00:00:00Z`,
      latestCommitDate: `${review.latestDate}T00:00:00Z`,
    })
  }

  return merged
}

function fetchMergedPullRequests() {
  const pullRequests = []
  let cursor = null

  do {
    const page = graphqlRequest(pullRequestsQuery, { cursor })
    const connection = page.repository.pullRequests
    pullRequests.push(...connection.nodes)
    cursor = connection.pageInfo.hasNextPage ? connection.pageInfo.endCursor : null
    if (cursor) {
      sleepSync(GH_PAGE_DELAY_MS)
    }
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

  let lastError

  for (let attempt = 1; attempt <= GH_MAX_ATTEMPTS; attempt += 1) {
    try {
      const output = runGh(args, {
        maxBuffer: 1024 * 1024 * 32,
        attempts: 1,
      })
      const payload = JSON.parse(output)
      if (payload.errors?.length) {
        const message = payload.errors.map(error => error.message).join('; ')
        const error = new Error(message)
        if (attempt < GH_MAX_ATTEMPTS && isRetryableGhError(error)) {
          const delay = GH_RETRY_BASE_MS * (2 ** (attempt - 1))
          console.warn(
            `GraphQL errors (attempt ${attempt}/${GH_MAX_ATTEMPTS}): ${message}; retrying in ${delay}ms`,
          )
          sleepSync(delay)
          lastError = error
          continue
        }

        throw error
      }

      return payload.data
    }
    catch (error) {
      lastError = error
      if (attempt >= GH_MAX_ATTEMPTS || !isRetryableGhError(error)) {
        throw error
      }

      const delay = GH_RETRY_BASE_MS * (2 ** (attempt - 1))
      const summary = errorText(error).split('\n').find(Boolean) || 'unknown error'
      console.warn(
        `GraphQL request failed (attempt ${attempt}/${GH_MAX_ATTEMPTS}): ${summary}; retrying in ${delay}ms`,
      )
      sleepSync(delay)
    }
  }

  throw lastError
}

function collectReviewStatsByContributorKey(pullRequests, range) {
  const stats = new Map()

  for (const pullRequest of pullRequests) {
    const reviewedInPr = new Set()

    for (const review of pullRequest.reviews?.nodes ?? []) {
      const reviewer = normalizeLogin(review.author?.login)
      const submittedAt = review.submittedAt?.slice(0, 10) ?? null

      if (
        !reviewer
        || isBotLogin(reviewer)
        || reviewedInPr.has(reviewer)
        || !reviewStates.has(review.state)
        || !submittedAt
        || !isDateInRange(submittedAt, range)
      ) {
        continue
      }

      reviewedInPr.add(reviewer)

      const key = resolveReviewerContributorKey(reviewer)
      if (!key) {
        continue
      }

      const login = key.slice('github:'.length)
      const current = stats.get(key) ?? {
        key,
        login,
        count: 0,
        latestDate: submittedAt,
      }
      current.count += 1
      current.latestDate = maxDate(current.latestDate, submittedAt)
      stats.set(key, current)
    }
  }

  return stats
}

function resolveReviewerContributorKey(login) {
  const normalized = normalizeLogin(login)
  if (!normalized || isBotLogin(normalized)) {
    return null
  }

  const override = overrideByLogin.get(normalized)
  const canonicalLogin = normalizeLogin(override?.login ?? login)

  return `github:${canonicalLogin}`
}

function isDateInRange(date, range) {
  return (!range.startDate || date >= range.startDate) && (!range.endDate || date <= range.endDate)
}

function isBotLogin(login) {
  const normalized = normalizeLogin(login)
  if (!normalized) {
    return true
  }

  return (
    normalized.includes('[bot]')
    || normalized.endsWith('-bot')
    || normalized === 'dependabot'
    || normalized === 'github-actions'
    || normalized.includes('copilot')
    || normalized.includes('codex-connector')
  )
}

function readContributorRows(startDate) {
  try {
    return readGitHubCommitRows(startDate)
  }
  catch (error) {
    console.warn(`Falling back to local git contributor data: ${error.message}`)

    return readGitLog(startDate)
  }
}

function readReleaseTimeline() {
  const releaseByTag = readGitHubReleaseMap()
  const v01 = releaseByTag.get(releaseTags.v01) ?? readLocalReleaseTagIfAvailable(releaseTags.v01)
  const v02 = releaseByTag.get(releaseTags.v02) ?? readLocalReleaseTagIfAvailable(releaseTags.v02)
  const v03 = releaseByTag.get(releaseTags.v03) ?? readLocalReleaseTagIfAvailable(releaseTags.v03)

  if (!v01 || !v02 || !v03) {
    return null
  }

  return { v01, v02, v03 }
}

function readGitHubReleaseMap() {
  try {
    const output = runGh(['api', `repos/${githubRepo}/releases`], {
      maxBuffer: 1024 * 1024 * 4,
    })

    const releases = uniqueReleasesByTag(JSON.parse(output)
      .filter(candidate => !candidate.draft)
      .sort((left, right) => String(right.published_at).localeCompare(String(left.published_at))))

    return new Map(releases
      .filter(release => release?.tag_name && release?.published_at)
      .map(release => [release.tag_name, {
        tagName: release.tag_name,
        name: release.name ?? release.tag_name,
        publishedAt: release.published_at,
      }]))
  }
  catch (error) {
    console.warn(`Falling back to local git release tag data: ${error.message}`)

    return new Map()
  }
}

function readLocalReleaseTagIfAvailable(tagName) {
  try {
    return readLocalReleaseTag(tagName)
  }
  catch (error) {
    console.warn(`Unable to read local release tag ${tagName}: ${error.message}`)

    return null
  }
}

function uniqueReleasesByTag(releases) {
  const seen = new Set()
  const unique = []

  for (const release of releases) {
    const tagName = release?.tag_name
    if (!tagName || seen.has(tagName)) {
      continue
    }

    seen.add(tagName)
    unique.push(release)
  }

  return unique
}

function readLocalReleaseTag(tagName) {
  const publishedAt = execFileSync('git', ['log', '-1', '--format=%aI', tagName], {
    cwd: repoRoot,
    encoding: 'utf8',
    stdio: ['ignore', 'pipe', 'pipe'],
  }).trim()

  return {
    tagName,
    name: tagName,
    publishedAt,
  }
}

function readCommitShasBetweenTags(baseTagName, targetTagName) {
  if (!baseTagName || !targetTagName) {
    return null
  }

  try {
    const output = runGh([
      'api',
      '--paginate',
      '--slurp',
      `repos/${githubRepo}/compare/${baseTagName}...${targetTagName}?per_page=100`,
    ], {
      maxBuffer: 1024 * 1024 * 64,
    })
    const pages = JSON.parse(output)
    const shas = pages.flatMap(page => page.commits ?? []).map(commit => commit.sha).filter(Boolean)

    if (shas.length > 0) {
      return new Set(shas)
    }
  }
  catch (error) {
    console.warn(`Falling back to local tag comparison: ${error.message}`)
  }

  try {
    execFileSync('git', ['rev-parse', '--verify', `${baseTagName}^{commit}`], {
      cwd: repoRoot,
      encoding: 'utf8',
      stdio: ['ignore', 'pipe', 'pipe'],
    })
    execFileSync('git', ['rev-parse', '--verify', `${targetTagName}^{commit}`], {
      cwd: repoRoot,
      encoding: 'utf8',
      stdio: ['ignore', 'pipe', 'pipe'],
    })

    const output = execFileSync('git', ['log', '--no-merges', '--format=%H', `${baseTagName}..${targetTagName}`], {
      cwd: repoRoot,
      encoding: 'utf8',
      maxBuffer: 1024 * 1024 * 8,
      stdio: ['ignore', 'pipe', 'pipe'],
    }).trim()

    return new Set(output ? output.split('\n') : [])
  }
  catch (error) {
    console.warn(`Falling back to release-date contributor detection: ${error.message}`)

    return null
  }
}

function readCommitShasSinceTag(tagName) {
  if (!tagName) {
    return null
  }

  try {
    const output = runGh([
      'api',
      '--paginate',
      '--slurp',
      `repos/${githubRepo}/compare/${tagName}...main?per_page=100`,
    ], {
      maxBuffer: 1024 * 1024 * 64,
    })
    const pages = JSON.parse(output)
    const shas = pages.flatMap(page => page.commits ?? []).map(commit => commit.sha).filter(Boolean)

    if (shas.length > 0) {
      return new Set(shas)
    }
  }
  catch (error) {
    console.warn(`Falling back to local tag comparison: ${error.message}`)
  }

  try {
    execFileSync('git', ['rev-parse', '--verify', `${tagName}^{commit}`], {
      cwd: repoRoot,
      encoding: 'utf8',
      stdio: ['ignore', 'pipe', 'pipe'],
    })

    const output = execFileSync('git', ['log', '--no-merges', '--format=%H', `${tagName}..HEAD`], {
      cwd: repoRoot,
      encoding: 'utf8',
      maxBuffer: 1024 * 1024 * 8,
      stdio: ['ignore', 'pipe', 'pipe'],
    }).trim()

    return new Set(output ? output.split('\n') : [])
  }
  catch (error) {
    console.warn(`Falling back to release-date contributor detection: ${error.message}`)

    return null
  }
}

function readCommitShasThroughTag(tagName) {
  if (!tagName) {
    return null
  }

  try {
    execFileSync('git', ['rev-parse', '--verify', `${tagName}^{commit}`], {
      cwd: repoRoot,
      encoding: 'utf8',
      stdio: ['ignore', 'pipe', 'pipe'],
    })

    const output = execFileSync('git', ['log', '--no-merges', '--format=%H', tagName], {
      cwd: repoRoot,
      encoding: 'utf8',
      maxBuffer: 1024 * 1024 * 8,
      stdio: ['ignore', 'pipe', 'pipe'],
    }).trim()

    return new Set(output ? output.split('\n') : [])
  }
  catch (error) {
    console.warn(`Falling back to release-date contributor detection: ${error.message}`)

    return null
  }
}

function readGitHubCommitRows(startDate) {
  const since = startDate ? `&since=${startDate}T00:00:00Z` : ''
  const endpoint = `repos/${githubRepo}/commits?sha=main&per_page=100${since}`
  const output = runGh(['api', '--paginate', '--slurp', endpoint], {
    maxBuffer: 1024 * 1024 * 64,
  })

  if (!output) {
    return []
  }

  const pages = JSON.parse(output)

  return pages
    .flat()
    .filter(commit => (commit.parents ?? []).length <= 1)
    .map((commit) => {
      const name = commit.commit?.author?.name ?? ''
      const email = commit.commit?.author?.email ?? ''
      const githubLogin = commit.author?.login
      const pullAuthor = githubLogin ? null : readAssociatedPullAuthorIfNeeded(commit.sha, name, email)

      return {
        sha: commit.sha,
        name,
        email,
        date: commit.commit?.author?.date ?? commit.commit?.committer?.date ?? '',
        githubLogin: githubLogin ?? pullAuthor?.login,
        githubLoginSource: githubLogin ? 'commit' : pullAuthor ? 'pull' : undefined,
        avatarUrl: commit.author?.avatar_url ?? pullAuthor?.avatarUrl,
      }
    })
    .filter(row => row.date)
}

function readGitLog(startDate) {
  const args = [
    'log',
    'HEAD',
    '--no-merges',
    '--format=%H%x1f%aN%x1f%aE%x1f%aI',
  ]

  if (startDate) {
    args.push(`--since=${startDate}`)
  }

  const output = execFileSync('git', args, {
    cwd: repoRoot,
    encoding: 'utf8',
    stdio: ['ignore', 'pipe', 'pipe'],
  }).trim()

  if (!output) {
    return []
  }

  return output.split('\n').map((line) => {
    const [sha, name, email, date] = line.split('\u001f')

    return { sha, name, email, date }
  })
}

function filterRowsByDateRange(rows, startDate, endDate) {
  return rows.filter((row) => {
    const date = row.date.slice(0, 10)

    return (!startDate || date >= startDate) && (!endDate || date <= endDate)
  })
}

function resolveIdentity(name, email, source = {}) {
  const normalizedEmail = normalizeEmail(email)
  const normalizedName = normalizeName(name)
  const githubLogin = source.githubLogin
  const avatarUrl = source.avatarUrl
  const isPullLogin = source.githubLoginSource === 'pull'
  const normalizedGithubLogin = normalizeLogin(githubLogin)
  const override = overrideByEmail.get(normalizedEmail) ?? overrideByName.get(normalizedName)

  if (isBotAuthor(normalizedName, normalizedEmail, normalizedGithubLogin)) {
    return { key: normalizedEmail || normalizedName || normalizedGithubLogin, name, isBot: true }
  }

  if (githubLogin && !isPullLogin) {
    const login = String(githubLogin)
    const override = overrideByLogin.get(normalizedGithubLogin)
    const profile = avatarUrl ? null : readGitHubUserProfile(login)

    return {
      key: `github:${normalizeLogin(profile?.login ?? login)}`,
      name: override?.name ?? name ?? login,
      login: profile?.login ?? login,
      avatarLogin: profile?.login ?? login,
      avatarUrl: avatarUrl ?? profile?.avatarUrl,
      avatarSeed: normalizeLogin(profile?.login ?? login),
      isBot: false,
    }
  }

  if (override) {
    const login = override.login ?? githubLogin
    const profile = login ? readGitHubUserProfile(login) : null

    return {
      key: login ? `github:${normalizeLogin(profile?.login ?? login)}` : override.key,
      name: override.name,
      login: profile?.login ?? login,
      avatarLogin: override.avatarLogin ?? profile?.login ?? login,
      avatarUrl: avatarUrl ?? profile?.avatarUrl,
      avatarSeed: override.key,
      isBot: false,
    }
  }

  const localGithubLogin = extractGithubLogin(normalizedEmail)
  const resolvedGithubLogin = localGithubLogin ?? githubLogin
  const profile = resolvedGithubLogin ? readGitHubUserProfile(resolvedGithubLogin) : null
  const login = profile?.login ?? resolvedGithubLogin
  const key = login ? `github:${normalizeLogin(login)}` : `email:${normalizedEmail || normalizedName}`

  return {
    key,
    name: name || profile?.name || login || 'Unknown contributor',
    login,
    avatarLogin: login,
    avatarUrl: avatarUrl ?? profile?.avatarUrl,
    avatarSeed: normalizedEmail || normalizedName || key,
    isBot: false,
  }
}

function readAssociatedPullAuthorIfNeeded(sha, name, email) {
  const normalizedEmail = normalizeEmail(email)
  const override = overrideByEmail.get(normalizedEmail) ?? overrideByName.get(normalizeName(name))

  if (override?.login || extractGithubLogin(normalizedEmail)) {
    return null
  }

  return readAssociatedPullAuthor(sha)
}

function readAssociatedPullAuthor(sha) {
  if (!sha) {
    return null
  }

  if (pullAuthorBySha.has(sha)) {
    return pullAuthorBySha.get(sha)
  }

  try {
    const output = runGh([
      'api',
      '-H',
      'Accept: application/vnd.github+json',
      `repos/${githubRepo}/commits/${sha}/pulls`,
    ], {
      maxBuffer: 1024 * 1024 * 2,
    })
    const pull = JSON.parse(output).find((candidate) => {
      const login = normalizeLogin(candidate?.user?.login)

      return login && login !== 'ghost'
    })
    const author = pull
      ? {
          login: pull.user.login,
          avatarUrl: pull.user.avatar_url,
        }
      : null

    pullAuthorBySha.set(sha, author)

    return author
  }
  catch {
    pullAuthorBySha.set(sha, null)

    return null
  }
}

function readGitHubUserProfile(login) {
  const normalizedLogin = normalizeLogin(login)

  if (!normalizedLogin) {
    return null
  }

  if (githubProfileByLogin.has(normalizedLogin)) {
    return githubProfileByLogin.get(normalizedLogin)
  }

  try {
    const output = runGh(['api', `users/${encodeURIComponent(String(login))}`], {
      maxBuffer: 1024 * 1024,
    })
    const user = JSON.parse(output)
    const profile = {
      login: user.login ?? login,
      name: user.name,
      avatarUrl: user.avatar_url,
    }

    githubProfileByLogin.set(normalizedLogin, profile)

    return profile
  }
  catch {
    githubProfileByLogin.set(normalizedLogin, null)

    return null
  }
}

function extractGithubLogin(email) {
  const match = email.match(/^\d+\+([^@]+)@users\.noreply\.github\.com$/i)

  return match?.[1]
}

function isBotAuthor(name, email, login = '') {
  return (
    name.includes('[bot]')
    || name === 'copilot'
    || login.includes('[bot]')
    || login === 'copilot'
    || email.includes('dependabot')
    || email.includes('copilot')
  )
}

function normalizeEmail(email) {
  return String(email ?? '').trim().toLowerCase()
}

function normalizeName(name) {
  return String(name ?? '').trim().toLowerCase()
}

function normalizeLogin(login) {
  return String(login ?? '').trim().toLowerCase()
}

function minDate(left, right) {
  return left < right ? left : right
}

function maxDate(left, right) {
  return left > right ? left : right
}

function formatLocalDate(date) {
  const year = date.getFullYear()
  const month = String(date.getMonth() + 1).padStart(2, '0')
  const day = String(date.getDate()).padStart(2, '0')

  return `${year}-${month}-${day}`
}

function renderTypeScript(generatedAt, snapshots, newContributorsSinceRelease) {
  return `/* eslint-disable */
// This file is generated by \`npm run contributors:rank\`.
// Source: GitHub commits API with local git fallback, plus merged PR reviews.

export type ContributorRankRange = 'v03ToNow' | 'v02ToV03' | 'v01ToV02' | 'v0ToV01' | 'all'

export interface ContributorRankEntry {
  rank: number
  key: string
  name: string
  login?: string
  avatarLogin?: string
  avatarUrl?: string
  avatarSeed: string
  commits: number
  reviews: number
  share: number
  firstCommitDate: string
  latestCommitDate: string
  isNewContributorSinceRelease?: boolean
}

export interface ContributorRankSnapshot {
  id: ContributorRankRange
  label: string
  generatedAt: string
  startDate: string | null
  endDate: string
  description: string
  totalCommits: number
  totalReviews: number
  totalContributors: number
  newContributors: number
  entries: ContributorRankEntry[]
}

export interface NewContributorsSinceReleaseSnapshot {
  tagName: string | null
  releaseName: string | null
  releaseDate: string | null
  targetTagName: string | null
  targetReleaseName: string | null
  targetReleaseDate: string | null
  comparisonMode: 'tag' | 'date' | 'none'
  totalCommits: number
  totalContributors: number
  entries: ContributorRankEntry[]
}

export const contributorRankGeneratedAt = '${generatedAt}'

export const newContributorsSinceRelease = ${JSON.stringify(newContributorsSinceRelease, null, 2)} satisfies NewContributorsSinceReleaseSnapshot

export const contributorRankData = ${JSON.stringify(snapshots, null, 2)} satisfies Record<ContributorRankRange, ContributorRankSnapshot>
`
}
