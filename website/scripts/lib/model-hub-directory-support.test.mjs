import assert from 'node:assert/strict'
import { readFileSync, readdirSync } from 'node:fs'
import { dirname, resolve } from 'node:path'
import { test } from 'node:test'
import { fileURLToPath } from 'node:url'
import ts from 'typescript'

const repositoryRoot = resolve(dirname(fileURLToPath(import.meta.url)), '../../..')
const helperPath = resolve(repositoryRoot, 'website/src/data/modelHubDirectorySupport.ts')
const helperJavaScript = ts.transpileModule(readFileSync(helperPath, 'utf8'), {
  compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2020 },
  fileName: helperPath,
}).outputText
const support = await import(
  `data:text/javascript;base64,${Buffer.from(helperJavaScript).toString('base64')}`,
)

test('directory reset tracks every user-facing filter', () => {
  const defaults = support.modelHubDirectoryDefaults
  const changes = {
    search: 'qwen',
    kind: 'virtual',
    distribution: 'open_weights',
    publisher: 'OpenAI',
    provider: 'vllm',
    capability: 'tools',
    lifecycle: 'all',
    sort: 'context',
  }

  assert.equal(support.modelHubActiveFilterCount(defaults), 0)
  for (const [field, value] of Object.entries(changes)) {
    assert.equal(
      support.modelHubActiveFilterCount({ ...defaults, [field]: value }),
      1,
      field,
    )
  }
})

test('model hub source avoids iterable spreads unsupported by the website build target', () => {
  const sourcePaths = [
    resolve(repositoryRoot, 'website/src/pages/models.tsx'),
    resolve(repositoryRoot, 'website/src/data/modelHubBenchmarkSupport.ts'),
    ...readdirSync(resolve(repositoryRoot, 'website/src/components/model-hub'))
      .filter(name => name.endsWith('.tsx'))
      .map(name => resolve(repositoryRoot, 'website/src/components/model-hub', name)),
  ]

  for (const sourcePath of sourcePaths) {
    const source = readFileSync(sourcePath, 'utf8')
    assert.doesNotMatch(source, /\[\s*\.\.\.(?:new\s+Set\b|[^\]]+\.values\(\)|[^\]]*querySelectorAll\()/s, sourcePath)
  }
})

test('model hub lives in the website More menu instead of consuming a primary nav slot', () => {
  const config = readFileSync(resolve(repositoryRoot, 'website/docusaurus.config.ts'), 'utf8')
  const moreMenu = config.slice(config.indexOf('label: \'More\''), config.indexOf('label: \'Dashboard\''))

  assert.match(moreMenu, /label: 'Model Hub'[\s\S]*?to: '\/models'/)
  assert.doesNotMatch(config, /label: 'Models'[\s\S]*?className: 'nav-primary'/)
})

test('website model filters and reset control remain visible', () => {
  const source = readFileSync(resolve(repositoryRoot, 'website/src/components/model-hub/ModelHubDirectory.tsx'), 'utf8')

  assert.match(source, /<div className=\{styles\.filterRow\} aria-label="Model filters">/)
  assert.match(source, /className=\{styles\.resetFilters\}[\s\S]*?onClick=\{resetFilters\}/)
})

test('website model table fills its container and falls back to contained horizontal scroll', () => {
  const css = readFileSync(
    resolve(repositoryRoot, 'website/src/components/model-hub/modelHubDirectory.module.css'),
    'utf8',
  )
  const frameRule = css.slice(css.indexOf('.tableFrame {'), css.indexOf('.tableFrame > table {'))
  const tableRule = css.slice(css.indexOf('.tableFrame > table {'), css.indexOf('.modelColumn {'))
  const cellRule = css.slice(css.indexOf('.tableFrame th,\n.tableFrame td {'), css.indexOf('.tableFrame th {'))

  // The frame owns the scrollbar so a narrow table never widens the page.
  assert.match(frameRule, /overflow-x:\s*auto/)
  assert.match(frameRule, /max-width:\s*100%/)

  // The table fills the frame when it fits, and floors at a readable
  // minimum (rather than shrinking indefinitely) so the frame's own
  // horizontal scroll is what reveals the rest, instead of clipped cells.
  assert.match(tableRule, /width:\s*100%/)
  assert.match(tableRule, /min-width:\s*(?!0\b)[\d.]+rem/)

  // Cells truncate cleanly with an ellipsis instead of wrapping and
  // growing the row when a column is narrower than its content.
  assert.match(cellRule, /white-space:\s*nowrap/)
  assert.match(cellRule, /text-overflow:\s*ellipsis/)
  assert.match(cellRule, /overflow:\s*hidden/)

  // The narrower breakpoints that hide columns also shrink the readable
  // minimum so the remaining columns can still fill without a stale,
  // wider floor forcing an unnecessary scrollbar.
  const narrowBreakpoint = css.slice(css.indexOf('@media (max-width: 780px)'), css.indexOf('@media (max-width: 640px)'))
  const narrowestBreakpoint = css.slice(css.indexOf('@media (max-width: 640px)'))
  assert.match(narrowBreakpoint, /\.tableFrame > table \{\s*min-width:\s*[\d.]+rem;/)
  assert.match(narrowestBreakpoint, /\.tableFrame > table \{\s*min-width:\s*[\d.]+rem;/)
})
