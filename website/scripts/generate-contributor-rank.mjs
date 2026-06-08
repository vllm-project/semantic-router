import { execFileSync } from 'node:child_process'
import { existsSync, mkdirSync, writeFileSync } from 'node:fs'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const scriptDir = dirname(fileURLToPath(import.meta.url))
const repoRoot = resolve(scriptDir, '..', '..')
const outputPath = resolve(repoRoot, 'website', 'src', 'data', 'contributorRank.generated.ts')
const githubRepo = 'vllm-project/semantic-router'

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

const rangeDefinitions = [
  {
    id: 'last3months',
    label: 'Last 3 months',
    startDate: formatLocalDate(shiftMonths(new Date(), -3)),
    description: 'Recent non-merge commit activity.',
  },
  {
    id: 'last365days',
    label: 'Last 365 days',
    startDate: formatLocalDate(shiftDays(new Date(), -365)),
    description: 'Trailing year non-merge commit activity.',
  },
  {
    id: 'all',
    label: 'All time',
    startDate: null,
    description: 'Full repository non-merge commit history.',
  },
]

try {
  const generatedAt = formatLocalDate(new Date())
  const allRows = readContributorRows(null)
  const releaseWindow = readReleaseWindow()
  const newContributorsSinceRelease = buildNewContributorsSinceRelease(releaseWindow, allRows)
  const newContributorKeys = new Set(newContributorsSinceRelease.entries.map(entry => entry.key))
  const snapshots = Object.fromEntries(
    rangeDefinitions.map(range => [range.id, buildSnapshot(range, generatedAt, allRows, newContributorKeys)]),
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

function buildSnapshot(range, generatedAt, allRows, newContributorKeys) {
  const rows = filterRowsByStartDate(allRows, range.startDate)
  const byContributor = collectContributorStats(rows)
  const totalCommits = [...byContributor.values()].reduce((sum, entry) => sum + entry.commits, 0)
  const entries = rankContributorEntries(byContributor, totalCommits, newContributorKeys)
  const newContributors = entries.filter(entry => entry.isNewContributorSinceRelease).length

  return {
    id: range.id,
    label: range.label,
    generatedAt,
    startDate: range.startDate,
    endDate: generatedAt,
    description: range.description,
    totalCommits,
    totalContributors: entries.length,
    newContributors,
    entries,
  }
}

function buildNewContributorsSinceRelease(releaseWindow, allRows) {
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
  const entries = rankContributorEntries(byContributor, totalCommits, new Set(byContributor.keys()))

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

function rankContributorEntries(byContributor, totalCommits, newContributorKeys) {
  return [...byContributor.values()]
    .sort((left, right) => {
      if (right.commits !== left.commits) {
        return right.commits - left.commits
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
      key: entry.key,
      commits: entry.commits,
      share: totalCommits > 0 ? Number((entry.commits / totalCommits).toFixed(4)) : 0,
      firstCommitDate: entry.firstCommitDate.slice(0, 10),
      latestCommitDate: entry.latestCommitDate.slice(0, 10),
      isNewContributorSinceRelease: newContributorKeys.has(entry.key),
    }))
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

function readReleaseWindow() {
  try {
    const output = execFileSync('gh', ['api', `repos/${githubRepo}/releases`], {
      cwd: repoRoot,
      encoding: 'utf8',
      maxBuffer: 1024 * 1024 * 4,
      stdio: ['ignore', 'pipe', 'pipe'],
    }).trim()

    const releases = uniqueReleasesByTag(JSON.parse(output)
      .filter(candidate => !candidate.draft)
      .sort((left, right) => String(right.published_at).localeCompare(String(left.published_at))))

    const [targetRelease, baseRelease] = releases
    if (baseRelease?.tag_name && baseRelease?.published_at && targetRelease?.tag_name && targetRelease?.published_at) {
      return {
        baseRelease: {
          tagName: baseRelease.tag_name,
          name: baseRelease.name ?? baseRelease.tag_name,
          publishedAt: baseRelease.published_at,
        },
        targetRelease: {
          tagName: targetRelease.tag_name,
          name: targetRelease.name ?? targetRelease.tag_name,
          publishedAt: targetRelease.published_at,
        },
      }
    }
  }
  catch (error) {
    console.warn(`Falling back to local git release tag data: ${error.message}`)
  }

  return readLocalReleaseWindow()
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

function readLocalReleaseWindow() {
  try {
    const tagNames = execFileSync('git', ['tag', '--sort=-creatordate'], {
      cwd: repoRoot,
      encoding: 'utf8',
      stdio: ['ignore', 'pipe', 'pipe'],
    }).trim().split('\n').filter(Boolean)

    const [targetTagName, baseTagName] = tagNames
    if (!targetTagName || !baseTagName) {
      return null
    }

    return {
      baseRelease: readLocalReleaseTag(baseTagName),
      targetRelease: readLocalReleaseTag(targetTagName),
    }
  }
  catch (error) {
    console.warn(`Unable to read local release tag data: ${error.message}`)

    return null
  }
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
    const output = execFileSync('gh', [
      'api',
      '--paginate',
      '--slurp',
      `repos/${githubRepo}/compare/${baseTagName}...${targetTagName}?per_page=100`,
    ], {
      cwd: repoRoot,
      encoding: 'utf8',
      maxBuffer: 1024 * 1024 * 64,
      stdio: ['ignore', 'pipe', 'pipe'],
    }).trim()
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

function readGitHubCommitRows(startDate) {
  const since = startDate ? `&since=${startDate}T00:00:00Z` : ''
  const endpoint = `repos/${githubRepo}/commits?sha=main&per_page=100${since}`
  const output = execFileSync('gh', ['api', '--paginate', '--slurp', endpoint], {
    cwd: repoRoot,
    encoding: 'utf8',
    maxBuffer: 1024 * 1024 * 64,
    stdio: ['ignore', 'pipe', 'pipe'],
  }).trim()

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

function filterRowsByStartDate(rows, startDate) {
  if (!startDate) {
    return rows
  }

  return rows.filter(row => row.date >= `${startDate}T00:00:00`)
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
    const output = execFileSync('gh', [
      'api',
      '-H',
      'Accept: application/vnd.github+json',
      `repos/${githubRepo}/commits/${sha}/pulls`,
    ], {
      cwd: repoRoot,
      encoding: 'utf8',
      maxBuffer: 1024 * 1024 * 2,
      stdio: ['ignore', 'pipe', 'ignore'],
    }).trim()
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
    const output = execFileSync('gh', ['api', `users/${encodeURIComponent(String(login))}`], {
      cwd: repoRoot,
      encoding: 'utf8',
      maxBuffer: 1024 * 1024,
      stdio: ['ignore', 'pipe', 'ignore'],
    }).trim()
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

function shiftMonths(date, months) {
  const next = new Date(date)
  next.setMonth(next.getMonth() + months)

  return next
}

function shiftDays(date, days) {
  const next = new Date(date)
  next.setDate(next.getDate() + days)

  return next
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
// Source: GitHub commits API with local git fallback.

export type ContributorRankRange = 'last3months' | 'last365days' | 'all'

export interface ContributorRankEntry {
  rank: number
  key: string
  name: string
  login?: string
  avatarLogin?: string
  avatarUrl?: string
  avatarSeed: string
  commits: number
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
