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

test('website model filters collapse without hiding the restore control', () => {
  const source = readFileSync(resolve(repositoryRoot, 'website/src/components/model-hub/ModelHubDirectory.tsx'), 'utf8')
  const stylesheet = readFileSync(resolve(repositoryRoot, 'website/src/components/model-hub/modelHubDirectory.module.css'), 'utf8')

  assert.match(source, /aria-label=\{filtersCollapsed \? 'Show model filters' : 'Hide model filters'\}/)
  assert.match(source, /aria-controls="website-model-hub-filter-controls"/)
  assert.match(stylesheet, /\.directoryCollapsed\s*\{\s*grid-template-columns: 2\.9rem minmax\(0, 1fr\)/)
  assert.match(stylesheet, /\.filterRailCollapsed > header\s*\{[\s\S]*?justify-content: center/)
})
