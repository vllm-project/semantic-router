import { test } from 'node:test'
import assert from 'node:assert/strict'
import { mergeEmailKeyedContributors } from './identity-merge.mjs'

function entriesToMap(entries) {
  return new Map(entries.map(entry => [entry.key, entry]))
}

test('merges the theohsiung email-keyed split from issue #2699 into the github-keyed entry', () => {
  // v03ToNow range from contributorRank.generated.ts: rank 2 vs rank 12.
  const byContributor = entriesToMap([
    {
      key: 'github:theohsiung',
      name: 'Theo Hsiung',
      login: 'theohsiung',
      commits: 34,
      firstCommitDate: '2025-11-03T08:00:00+08:00',
      latestCommitDate: '2026-01-20T08:00:00+08:00',
      emails: new Set(['139082137+theohsiung@users.noreply.github.com']),
      names: new Set(['Theo Hsiung']),
    },
    {
      key: 'email:theobear870924@gmail.com',
      name: 'theohsiung',
      commits: 5,
      firstCommitDate: '2025-10-28T08:00:00+08:00',
      latestCommitDate: '2025-12-01T08:00:00+08:00',
      emails: new Set(['theobear870924@gmail.com']),
      names: new Set(['theohsiung']),
    },
  ])

  const merged = mergeEmailKeyedContributors(byContributor)

  assert.equal(merged.size, 1)
  assert.equal(merged.has('email:theobear870924@gmail.com'), false)

  const entry = merged.get('github:theohsiung')

  assert.equal(entry.name, 'Theo Hsiung')
  assert.equal(entry.login, 'theohsiung')
  assert.equal(entry.commits, 39)
  assert.equal(entry.firstCommitDate, '2025-10-28T08:00:00+08:00')
  assert.equal(entry.latestCommitDate, '2026-01-20T08:00:00+08:00')
})

test('keeps entries separate when only a shared display name links them', () => {
  // Author names are user-controlled and collide (two authors committing as
  // "root"), so a shared display name alone is deliberately NOT merge
  // evidence; only name-equals-login is. The original #2699 splits have
  // self-healed upstream.
  const byContributor = entriesToMap([
    {
      key: 'github:wilsonwu',
      name: 'Wilson Wu',
      login: 'wilsonwu',
      commits: 12,
      firstCommitDate: '2025-09-01T00:00:00Z',
      latestCommitDate: '2026-01-10T00:00:00Z',
      emails: new Set(['wilsonwu@users.noreply.github.com']),
      names: new Set(['Wilson Wu']),
    },
    {
      key: 'email:iwilsonwu@gmail.com',
      name: 'Wilson Wu',
      commits: 3,
      firstCommitDate: '2025-12-05T00:00:00Z',
      latestCommitDate: '2026-02-01T00:00:00Z',
      emails: new Set(['iwilsonwu@gmail.com']),
      names: new Set(['Wilson Wu']),
    },
  ])

  const merged = mergeEmailKeyedContributors(byContributor)

  assert.equal(merged.size, 2)
  assert.equal(merged.get('github:wilsonwu').commits, 12)
  assert.equal(merged.get('email:iwilsonwu@gmail.com').commits, 3)
})

test('keeps entries separate when only shared author-email evidence links them', () => {
  // Email-evidence matching was dropped along with name matching: the
  // name-equals-login path is the only merge signal.
  const byContributor = entriesToMap([
    {
      key: 'github:someone',
      name: 'Someone',
      login: 'someone',
      commits: 7,
      firstCommitDate: '2025-01-01T00:00:00Z',
      latestCommitDate: '2025-06-01T00:00:00Z',
      emails: new Set(['personal@example.com']),
      names: new Set(['S. One']),
    },
    {
      key: 'email:personal@example.com',
      name: 'Different Display Name',
      commits: 2,
      firstCommitDate: '2025-03-01T00:00:00Z',
      latestCommitDate: '2025-07-01T00:00:00Z',
      emails: new Set(['personal@example.com']),
      names: new Set(['Different Display Name']),
    },
  ])

  const merged = mergeEmailKeyedContributors(byContributor)

  assert.equal(merged.size, 2)
  assert.equal(merged.get('github:someone').commits, 7)
  assert.equal(merged.get('email:personal@example.com').commits, 2)
})

test('keeps an email-keyed entry with no linking evidence as its own entry', () => {
  const byContributor = entriesToMap([
    {
      key: 'github:someone',
      name: 'Someone',
      login: 'someone',
      commits: 7,
      firstCommitDate: '2025-01-01T00:00:00Z',
      latestCommitDate: '2025-06-01T00:00:00Z',
      emails: new Set(['someone@example.com']),
      names: new Set(['Someone']),
    },
    {
      key: 'email:stranger@example.com',
      name: 'Stranger',
      commits: 4,
      firstCommitDate: '2025-02-01T00:00:00Z',
      latestCommitDate: '2025-05-01T00:00:00Z',
      emails: new Set(['stranger@example.com']),
      names: new Set(['Stranger']),
    },
  ])

  const merged = mergeEmailKeyedContributors(byContributor)

  assert.equal(merged.size, 2)
  assert.equal(merged.get('github:someone').commits, 7)
  assert.equal(merged.get('email:stranger@example.com').commits, 4)
})

test('does not merge when the linking evidence is ambiguous across github entries', () => {
  const byContributor = entriesToMap([
    {
      key: 'github:alpha',
      name: 'Jay Chen',
      login: 'alpha',
      commits: 5,
      firstCommitDate: '2025-01-01T00:00:00Z',
      latestCommitDate: '2025-06-01T00:00:00Z',
      emails: new Set(['alpha@example.com']),
      names: new Set(['Jay Chen']),
    },
    {
      key: 'github:beta',
      name: 'Jay Chen',
      login: 'beta',
      commits: 6,
      firstCommitDate: '2025-01-01T00:00:00Z',
      latestCommitDate: '2025-06-01T00:00:00Z',
      emails: new Set(['beta@example.com']),
      names: new Set(['Jay Chen']),
    },
    {
      key: 'email:jay@example.com',
      name: 'Jay Chen',
      commits: 2,
      firstCommitDate: '2025-02-01T00:00:00Z',
      latestCommitDate: '2025-03-01T00:00:00Z',
      emails: new Set(['jay@example.com']),
      names: new Set(['Jay Chen']),
    },
  ])

  const merged = mergeEmailKeyedContributors(byContributor)

  assert.equal(merged.size, 3)
  assert.equal(merged.get('github:alpha').commits, 5)
  assert.equal(merged.get('github:beta').commits, 6)
  assert.equal(merged.get('email:jay@example.com').commits, 2)
})

test('matches evidence case-insensitively', () => {
  const byContributor = entriesToMap([
    {
      key: 'github:theohsiung',
      name: 'Theo Hsiung',
      login: 'theohsiung',
      commits: 34,
      firstCommitDate: '2025-11-03T00:00:00Z',
      latestCommitDate: '2026-01-20T00:00:00Z',
      emails: new Set(['139082137+theohsiung@users.noreply.github.com']),
      names: new Set(['Theo Hsiung']),
    },
    {
      key: 'email:theobear870924@gmail.com',
      name: 'TheoHsiung',
      commits: 5,
      firstCommitDate: '2025-10-28T00:00:00Z',
      latestCommitDate: '2025-12-01T00:00:00Z',
      emails: new Set(['TheoBear870924@Gmail.com']),
      names: new Set(['TheoHsiung']),
    },
  ])

  const merged = mergeEmailKeyedContributors(byContributor)

  assert.equal(merged.size, 1)
  assert.equal(merged.get('github:theohsiung').commits, 39)
})

test('does not merge email-keyed entries into bot entries or into each other', () => {
  const byContributor = entriesToMap([
    {
      key: 'dependabot[bot]',
      name: 'dependabot[bot]',
      commits: 20,
      isBot: true,
      firstCommitDate: '2025-01-01T00:00:00Z',
      latestCommitDate: '2025-06-01T00:00:00Z',
      emails: new Set(['dependabot@example.com']),
      names: new Set(['dependabot[bot]']),
    },
    {
      key: 'email:first@example.com',
      name: 'Same Name',
      commits: 2,
      firstCommitDate: '2025-02-01T00:00:00Z',
      latestCommitDate: '2025-03-01T00:00:00Z',
      emails: new Set(['first@example.com']),
      names: new Set(['Same Name']),
    },
    {
      key: 'email:second@example.com',
      name: 'Same Name',
      commits: 3,
      firstCommitDate: '2025-02-01T00:00:00Z',
      latestCommitDate: '2025-03-01T00:00:00Z',
      emails: new Set(['second@example.com']),
      names: new Set(['Same Name']),
    },
  ])

  const merged = mergeEmailKeyedContributors(byContributor)

  assert.equal(merged.size, 3)
})

test('does not mutate the input map or its entries', () => {
  const emailEntry = {
    key: 'email:theobear870924@gmail.com',
    name: 'theohsiung',
    commits: 5,
    firstCommitDate: '2025-10-28T00:00:00Z',
    latestCommitDate: '2025-12-01T00:00:00Z',
    emails: new Set(['theobear870924@gmail.com']),
    names: new Set(['theohsiung']),
  }
  const githubEntry = {
    key: 'github:theohsiung',
    name: 'Theo Hsiung',
    login: 'theohsiung',
    commits: 34,
    firstCommitDate: '2025-11-03T00:00:00Z',
    latestCommitDate: '2026-01-20T00:00:00Z',
    emails: new Set(['139082137+theohsiung@users.noreply.github.com']),
    names: new Set(['Theo Hsiung']),
  }
  const byContributor = entriesToMap([githubEntry, emailEntry])

  mergeEmailKeyedContributors(byContributor)

  assert.equal(byContributor.size, 2)
  assert.equal(githubEntry.commits, 34)
  assert.equal(emailEntry.commits, 5)
})
