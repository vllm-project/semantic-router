import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { dirname, resolve } from 'node:path'
import { test } from 'node:test'
import { fileURLToPath } from 'node:url'
import ts from 'typescript'

const repositoryRoot = resolve(
  dirname(fileURLToPath(import.meta.url)),
  '../../..',
)
const helperPath = resolve(
  repositoryRoot,
  'website/src/data/modelHubEvaluationLabel.ts',
)
const helperSource = readFileSync(helperPath, 'utf8')
const helperJavaScript = ts.transpileModule(helperSource, {
  compilerOptions: {
    module: ts.ModuleKind.ESNext,
    target: ts.ScriptTarget.ES2020,
  },
  fileName: helperPath,
}).outputText
const { modelHubEvaluationConditionLabel } = await import(
  `data:text/javascript;base64,${Buffer.from(helperJavaScript).toString('base64')}`,
)

test('model hub distinguishes configurable effort from published run conditions', () => {
  const configurable = { reasoning_family: 'openai-reasoning' }
  const published = {}

  assert.equal(
    modelHubEvaluationConditionLabel(configurable, 'high'),
    'high effort',
  )
  assert.equal(
    modelHubEvaluationConditionLabel(published, 'unspecified'),
    'Effort not reported',
  )
  assert.equal(
    modelHubEvaluationConditionLabel(published, 'default'),
    'Published default',
  )
  assert.equal(
    modelHubEvaluationConditionLabel(published, 'enabled'),
    'Reasoning enabled',
  )
  assert.equal(
    modelHubEvaluationConditionLabel(published, 'disabled'),
    'Non-reasoning run',
  )
  assert.equal(
    modelHubEvaluationConditionLabel(published, 'adaptive'),
    'Adaptive reasoning',
  )
  assert.equal(
    modelHubEvaluationConditionLabel(published, 'none'),
    'No reasoning',
  )
  assert.equal(
    modelHubEvaluationConditionLabel(published, 'no_think'),
    'No reasoning',
  )
  assert.equal(
    modelHubEvaluationConditionLabel(published, 'custom_profile'),
    'Custom profile run',
  )
})

test('website benchmark bars and detail evidence use the shared condition label', () => {
  const page = readFileSync(
    resolve(repositoryRoot, 'website/src/pages/models.tsx'),
    'utf8',
  )

  assert.equal(page.match(/modelHubEvaluationConditionLabel\(/g)?.length, 2)
  assert.match(page, /aria-label=\{`Rank \$\{rank\}[^`]+\$\{conditionLabel\}/)
  assert.doesNotMatch(page, /readable\([^)]*reasoning_effort[^)]*\)/)
  assert.match(page, /if \(!value\) return 'Not published'/)
  assert.doesNotMatch(page, /Release date unavailable/)
})
