import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { test } from 'node:test'

const repositoryRoot = resolve(
  dirname(fileURLToPath(import.meta.url)),
  '../../..',
)
const searchBarPath = resolve(
  repositoryRoot,
  'website/src/theme/SearchBar/index.tsx',
)
const zhHansCatalogPath = resolve(
  repositoryRoot,
  'website/i18n/zh-Hans/code.json',
)

test('zh-Hans catalog covers every search palette translation ID', () => {
  const source = readFileSync(searchBarPath, 'utf8')
  const catalog = JSON.parse(readFileSync(zhHansCatalogPath, 'utf8'))
  const ids = [...new Set(
    [...source.matchAll(/theme\.SearchBar\.[A-Za-z0-9.]+/g)].map(match => match[0]),
  )]

  assert.ok(ids.length >= 17, `expected at least 17 palette IDs, found ${ids.length}`)
  for (const id of ids) {
    const entry = catalog[id]
    assert.ok(entry, `${id} is missing from website/i18n/zh-Hans/code.json`)
    assert.equal(typeof entry.message, 'string', `${id} has no message`)
    assert.notEqual(entry.message.trim(), '', `${id} has an empty message`)
  }
})

test('search palette keeps the trigger and empty-result text translatable', () => {
  const source = readFileSync(searchBarPath, 'utf8')

  assert.doesNotMatch(
    source,
    /site-search-trigger__label">Search</,
    'the trigger label must come from theme.SearchBar.triggerLabel',
  )
  assert.doesNotMatch(
    source,
    /^\s*No results for “/m,
    'the empty-result headline must come from theme.SearchBar.noResults',
  )
  assert.match(source, /id: 'theme\.SearchBar\.noResults'/)
})

test('zh-Hans no-results message keeps the query interpolation contract', () => {
  const source = readFileSync(searchBarPath, 'utf8')
  const catalog = JSON.parse(readFileSync(zhHansCatalogPath, 'utf8'))
  const message = catalog['theme.SearchBar.noResults']?.message ?? ''

  assert.match(
    message,
    /\{term\}/,
    'the zh-Hans noResults message must keep the {term} placeholder',
  )
  assert.match(
    source,
    /id: 'theme\.SearchBar\.noResults'[\s\S]*?\}\s*,\s*\{\s*term\s*\}\s*,?\s*\)/,
    'the noResults translation must receive the query as { term }',
  )
})
