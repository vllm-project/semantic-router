import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { test } from 'node:test'

const repositoryRoot = resolve(
  dirname(fileURLToPath(import.meta.url)),
  '../../..',
)
const communityLayoutPath = resolve(
  repositoryRoot,
  'website/src/components/community/CommunityLayout/index.tsx',
)
const teamPagePath = resolve(repositoryRoot, 'website/src/pages/community/team.tsx')
const zhHansCatalogPath = resolve(
  repositoryRoot,
  'website/i18n/zh-Hans/code.json',
)

const communityLayoutSource = readFileSync(communityLayoutPath, 'utf8')
const teamPageSource = readFileSync(teamPagePath, 'utf8')
const zhHansCatalog = JSON.parse(readFileSync(zhHansCatalogPath, 'utf8'))

/* Collapse the JSX/TS whitespace so a wrapped English literal compares equal to
 * the same text written on one line. */
function normalizeMessage(text) {
  return text
    .replace(/\{' '\}/g, ' ')
    .replace(/^['"`]|['"`]$/g, '')
    .replace(/\s+/g, ' ')
    .trim()
}

/* Collect the English default for every shell translation ID the Team page and
 * the shared navigation declare, in both the imperative and component forms. */
function declaredTranslations(source) {
  const found = new Map()

  for (const match of source.matchAll(
    /id:\s*'([a-z][A-Za-z0-9.]*)',\s*message:\s*'([^']*)'/g,
  )) {
    found.set(match[1], normalizeMessage(match[2]))
  }

  for (const match of source.matchAll(
    /<Translate\s+id="([a-z][A-Za-z0-9.]*)"[^>]*>([\s\S]*?)<\/Translate>/g,
  )) {
    found.set(match[1], normalizeMessage(match[2]))
  }

  for (const match of source.matchAll(
    /labelId:\s*'([a-z][A-Za-z0-9.]*)',\s*label:\s*'([^']*)'/g,
  )) {
    found.set(match[1], normalizeMessage(match[2]))
  }

  return found
}

const declared = new Map([
  ...declaredTranslations(communityLayoutSource),
  ...declaredTranslations(teamPageSource),
])

test('Team shell and shared navigation declare their translation IDs', () => {
  assert.ok(
    declared.size >= 19,
    `expected at least 19 community shell IDs, found ${declared.size}`,
  )

  for (const id of [
    'community.layout.eyebrow',
    'community.layout.sidebarLabel',
    'community.nav.group.people',
    'community.nav.group.howWeWork',
    'community.nav.item.codeOfConduct',
    'community.team.pageTitle',
    'community.team.pageDescription',
    'community.team.title',
    'community.team.description',
    'community.team.maintainers.title',
    'community.team.maintainers.description',
    'community.team.committers.title',
    'community.team.committers.description',
    'community.team.emeritus.title',
    'community.team.emeritus.description',
  ]) {
    assert.ok(declared.has(id), `${id} is not declared in the Team shell or navigation`)
  }
})

test('every declared ID has a zh-Hans entry translated away from the English default', () => {
  for (const [id, english] of declared) {
    const entry = zhHansCatalog[id]
    assert.ok(entry, `${id} is missing from website/i18n/zh-Hans/code.json`)
    assert.equal(typeof entry.message, 'string', `${id} has no message`)
    assert.notEqual(entry.message.trim(), '', `${id} has an empty message`)
    assert.notEqual(
      entry.message.trim(),
      english,
      `${id} still carries the English default: ${english}`,
    )
    assert.match(
      entry.message,
      /[\u4E00-\u9FFF]/,
      `${id} has no Chinese text: ${entry.message}`,
    )
  }
})

test('the emeritus roster keeps its interpolated date placeholders', () => {
  const message = zhHansCatalog['community.team.emeritus.description']?.message ?? ''

  assert.match(message, /\{fromDate\}/, 'the zh-Hans emeritus text must keep {fromDate}')
  assert.match(message, /\{toDate\}/, 'the zh-Hans emeritus text must keep {toDate}')
  assert.match(
    teamPageSource,
    /id="community\.team\.emeritus\.description"[\s\S]*?values=\{\{[\s\S]*?fromDate:[\s\S]*?toDate:/,
    'the emeritus description must pass both dates as interpolation values',
  )
  assert.match(
    teamPageSource,
    /<time dateTime=\{committerActivityWindow\.cutoffDate\}>/,
    'the roster window start must stay in a <time> element',
  )
  assert.match(
    teamPageSource,
    /<time dateTime=\{committerActivityWindow\.generatedAt\}>/,
    'the roster window end must stay in a <time> element',
  )
})

test('shared navigation keeps literal keys and destinations', () => {
  const routePairs = [
    ['team', '/community/team'],
    ['steering-committee', '/community/steering-committee'],
    ['work-groups', '/community/work-groups'],
    ['leaderboard', '/community/contributors'],
    ['governance', '/community/governance'],
    ['contributing', '/community/contributing'],
    ['code-of-conduct', '/community/code-of-conduct'],
  ]

  for (const [key, to] of routePairs) {
    assert.match(
      communityLayoutSource,
      new RegExp(`key: '${key}'[^}]*to: '${to}'`),
      `navigation entry ${key} must keep destination ${to}`,
    )
  }

  assert.match(
    communityLayoutSource,
    /function localizedNavGroups\(\): BrowseNavGroup\[\]/,
    'the rail labels must be resolved through translate()',
  )
  assert.match(
    communityLayoutSource,
    /groups=\{localizedNavGroups\(\)\}/,
    'the rail must receive the localized labels, not the untranslated data',
  )
})

test('Team page keeps no English shell literals', () => {
  const shellLiterals = [
    'title="Project Team"',
    'title="Maintainers"',
    'title="Committers"',
    'title="Emeritus Committers"',
    'description="The people responsible for the health',
    'description="Project-wide owners',
    'description="Active trusted contributors',
  ]

  for (const literal of shellLiterals) {
    assert.ok(
      !teamPageSource.includes(literal),
      `Team page still renders the English literal: ${literal}`,
    )
  }

  assert.match(
    teamPageSource,
    /title=\{translate\(\{ id: 'community\.team\.pageTitle'/,
    'the document title metadata must come from a translation ID',
  )
  assert.match(
    teamPageSource,
    /description=\{translate\(\{\s*id: 'community\.team\.pageDescription'/,
    'the document description metadata must come from a translation ID',
  )
})
