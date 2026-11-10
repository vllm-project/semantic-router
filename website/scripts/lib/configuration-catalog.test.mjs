import assert from 'node:assert/strict'
import { mkdtempSync, mkdirSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { test } from 'node:test'
import {
  extractOverviewGoal,
  findSingleTutorial,
  GENERATED_CATALOG_END,
  GENERATED_CATALOG_START,
  replaceGeneratedCatalog,
  renderConfigurationCatalog,
  signalTutorialType,
} from './configuration-catalog.mjs'

test('extracts the first sentence from the Overview prose', () => {
  const markdown = [
    '# Keyword Signal',
    '',
    '## Overview',
    '',
    '`keyword` matches explicit lexical patterns in the request. It maps to a fragment.',
    '',
    '## Configuration',
    '',
    'Example.',
  ].join('\n')

  assert.equal(
    extractOverviewGoal(markdown, 'keyword.md'),
    '`keyword` matches explicit lexical patterns in the request.',
  )
})

test('requires exactly one Overview section', () => {
  assert.throws(
    () => extractOverviewGoal('# Guide\n\nNo overview.\n', 'guide.md'),
    /exactly one "## Overview" heading; found 0/,
  )
  assert.throws(
    () => extractOverviewGoal('## Overview\n\nFirst.\n\n## Overview\n\nSecond.\n', 'guide.md'),
    /exactly one "## Overview" heading; found 2/,
  )
})

test('rejects missing and ambiguous tutorial matches', () => {
  const root = mkdtempSync(join(tmpdir(), 'configuration-catalog-'))
  mkdirSync(join(root, 'heuristic'), { recursive: true })
  mkdirSync(join(root, 'application'), { recursive: true })

  assert.throws(() => findSingleTutorial(root, 'keyword'), /No tutorial named keyword\.md/)

  writeFileSync(join(root, 'heuristic', 'keyword.md'), '# Keyword\n')
  assert.equal(findSingleTutorial(root, 'keyword'), join(root, 'heuristic', 'keyword.md'))

  writeFileSync(join(root, 'application', 'keyword.md'), '# Duplicate\n')
  assert.throws(() => findSingleTutorial(root, 'keyword'), /Ambiguous tutorial for keyword/)
})

test('types a signal by the layer its tutorial lives in', () => {
  const root = join(tmpdir(), 'signal')

  assert.equal(signalTutorialType(root, join(root, 'systemone.md')), 'primitive signal')
  assert.equal(signalTutorialType(root, join(root, 'application', 'pii.md')), 'application signal')
  assert.equal(signalTutorialType(root, join(root, 'heuristic', 'keyword.md')), 'heuristic signal')
  assert.throws(
    () => signalTutorialType(root, join(root, 'learned', 'pii.md')),
    /must be signal\/systemone\.md or live directly under signal\/application or signal\/heuristic/,
  )
  assert.throws(
    () => signalTutorialType(root, join(root, 'pii.md')),
    /must be signal\/systemone\.md/,
  )
})

test('replaces only the marked generated catalog block', () => {
  const page = `Before\n\n${GENERATED_CATALOG_START}\nold\n${GENERATED_CATALOG_END}\n\nAfter\n`
  const generated = `${GENERATED_CATALOG_START}\nnew\n${GENERATED_CATALOG_END}`

  assert.equal(
    replaceGeneratedCatalog(page, generated),
    `Before\n\n${generated}\n\nAfter\n`,
  )
})

test('publishes selectors and executors as separate tables', () => {
  const entry = {
    family: 'escalate',
    type: 'executor',
    goal: 'Call the candidates in order until one answer is accepted.',
    fragmentPath: 'config/fragments/algorithm/looper/escalate.yaml',
    fragmentKind: 'file',
    guide: '../tutorials/algorithm/looper/escalate',
  }
  const catalog = renderConfigurationCatalog({
    signals: [], selectors: [], executors: [entry], plugins: [],
  })
  assert.match(catalog, /### Selectors/)
  assert.match(catalog, /### Executors\n\n.*\n.*\n\| `escalate` — executor/)
  assert.doesNotMatch(catalog, /Native System One/)
})
