import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import Module, { createRequire } from 'node:module'
import { dirname, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { test } from 'node:test'

const repositoryRoot = resolve(
  dirname(fileURLToPath(import.meta.url)),
  '../../..',
)
const websiteRoot = join(repositoryRoot, 'website')
const communityLayoutPath = join(
  websiteRoot,
  'src/components/community/CommunityLayout/index.tsx',
)
const teamPagePath = join(websiteRoot, 'src/pages/community/team.tsx')
const zhHansCatalogPath = join(websiteRoot, 'i18n/zh-Hans/code.json')

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

  return found
}

const declared = new Map([
  ...declaredTranslations(communityLayoutSource),
  ...declaredTranslations(teamPageSource),
])

/* Render the real component against the real zh-Hans catalog, with the
 * Docusaurus runtime and the browse shell stubbed to their contracts. The
 * shell stub renders exactly the strings it is handed, so the assertion is
 * about what this component passes into the rail rather than about a regex. */
async function renderCommunityRail() {
  const require = createRequire(join(websiteRoot, 'package.json'))
  const React = require('react')
  const { renderToStaticMarkup } = require('react-dom/server')
  const babel = require('@babel/core')

  const source = readFileSync(communityLayoutPath, 'utf8')
  const { code } = babel.transformSync(source, {
    filename: communityLayoutPath,
    presets: [
      [require.resolve('@babel/preset-typescript'), { isTSX: true, allExtensions: true }],
      [require.resolve('@babel/preset-react'), { runtime: 'classic' }],
    ],
    plugins: [require.resolve('@babel/plugin-transform-modules-commonjs')],
    babelrc: false,
    configFile: false,
  })

  const translate = ({ id, message }) => zhHansCatalog[id]?.message ?? message ?? id
  const Translate = ({ id, children }) => translate({ id, message: children })

  const BrowseLayout = ({ groups, sidebarLabel }) => React.createElement(
    'nav',
    { 'aria-label': sidebarLabel },
    groups.map(group => React.createElement(
      'div',
      { key: group.key },
      React.createElement('span', { 'data-group-label': '' }, group.label),
      group.items.map(item => React.createElement(
        'a',
        { 'key': item.key, 'href': item.to, 'data-nav-item': item.key },
        item.label,
      )),
    )),
  )

  const stubs = {
    'react': React,
    '@docusaurus/Translate': Object.assign(Translate, { translate, default: Translate }),
    '@site/src/components/site/BrowseLayout': BrowseLayout,
  }

  const originalLoad = Module._load
  Module._load = function (request, parent, isMain) {
    if (request in stubs) return stubs[request]
    return originalLoad.call(this, request, parent, isMain)
  }

  try {
    const compiled = new Module(communityLayoutPath)
    compiled.filename = communityLayoutPath
    compiled.paths = Module._nodeModulePaths(dirname(communityLayoutPath))
    compiled._compile(code, communityLayoutPath)

    const CommunityLayout = compiled.exports.default
    const html = renderToStaticMarkup(
      React.createElement(CommunityLayout, { activeKey: 'team', title: 'Team' }, 'body'),
    )

    return {
      html,
      groupLabels: [...html.matchAll(/data-group-label[^>]*>([^<]*)</g)].map(match => match[1]),
      items: [...html.matchAll(/data-nav-item="([^"]*)"[^>]*>([^<]*)</g)]
        .map(match => ({ key: match[1], label: match[2] })),
      hrefs: [...html.matchAll(/href="([^"]*)"/g)].map(match => match[1]),
      sidebarLabel: (html.match(/aria-label="([^"]*)"/) ?? [])[1],
    }
  }
  finally {
    Module._load = originalLoad
  }
}

const NAV_ITEMS = [
  ['team', '/community/team', '开源团队'],
  ['steering-committee', '/community/steering-committee', '指导委员会'],
  ['work-groups', '/community/work-groups', '工作组'],
  ['leaderboard', '/community/contributors', '排行榜'],
  ['governance', '/community/governance', '治理'],
  ['contributing', '/community/contributing', '贡献指南'],
  ['code-of-conduct', '/community/code-of-conduct', '行为准则'],
]

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

test('the rendered rail shows the Chinese group headings and link labels', async () => {
  const rail = await renderCommunityRail()

  assert.deepEqual(
    rail.groupLabels,
    ['人员', '协作方式'],
    'the rail must render the zh-Hans group headings',
  )

  assert.deepEqual(
    rail.items.map(item => item.key),
    NAV_ITEMS.map(([key]) => key),
    'the rail must keep every navigation entry in order',
  )

  for (const [key, , label] of NAV_ITEMS) {
    const item = rail.items.find(candidate => candidate.key === key)
    assert.ok(item, `${key} is missing from the rendered rail`)
    assert.equal(item.label, label, `${key} must render its zh-Hans label`)
  }

  assert.equal(rail.sidebarLabel, '社区栏目', 'the rail must render its zh-Hans accessible label')
})

test('the rendered rail keeps every destination unchanged', async () => {
  const rail = await renderCommunityRail()

  assert.deepEqual(
    rail.hrefs,
    NAV_ITEMS.map(([, to]) => to),
    'every rendered link must keep its original destination',
  )
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
  for (const [key, to] of NAV_ITEMS) {
    assert.ok(
      communityLayoutSource.includes(`key: '${key}'`),
      `navigation entry ${key} must keep its key`,
    )
    assert.ok(
      communityLayoutSource.includes(`to: '${to}'`),
      `navigation entry ${key} must keep destination ${to}`,
    )
  }

  assert.doesNotMatch(
    communityLayoutSource,
    /label:\s*(?:translate\(\{\s*id:\s*)?group\.label/,
    'the rail labels must be literal translate() calls so the extractor can read them',
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
