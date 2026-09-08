import assert from 'node:assert/strict'
import { existsSync, readFileSync } from 'node:fs'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { test } from 'node:test'

const repositoryRoot = resolve(
  dirname(fileURLToPath(import.meta.url)),
  '../../..',
)
const catalog = JSON.parse(
  readFileSync(
    resolve(repositoryRoot, 'website/static/model-catalog/catalog.json'),
    'utf8',
  ),
)

test('built-in chat models publish the routing limits needed by model cards', () => {
  const physicalChatModels = catalog.models.filter(
    model => model.kind === 'physical' && model.capabilities.includes('chat'),
  )
  const openWeightModels = physicalChatModels.filter(
    model => model.distribution.type === 'open_weights',
  )
  const gemini = physicalChatModels.find(
    model => model.id === 'google/gemini-3.8-flash',
  )

  assert.ok(
    physicalChatModels.every(model => model.limits?.context_window_size > 0),
  )
  assert.ok(openWeightModels.every(model => model.parameter_size))
  assert.equal(gemini?.limits?.context_window_size, 1_048_576)
  assert.equal(gemini?.limits?.max_output_tokens, 65_536)
})

test('website catalog exposes every model access relationship explicitly', () => {
  const allowed = new Set([
    'first_party',
    'managed_cloud',
    'gateway',
    'self_hosted',
  ])
  const bindings = catalog.providers.flatMap(
    provider => provider.models ?? [],
  )

  assert.ok(bindings.length > 0)
  assert.ok(bindings.every(binding => allowed.has(binding.relationship)))
  assert.deepEqual(
    new Set(bindings.map(binding => binding.relationship)),
    allowed,
  )
})

test('model details present human-readable access relationship labels', () => {
  const page = readFileSync(
    resolve(repositoryRoot, 'website/src/components/model-hub/ModelHubDetail.tsx'),
    'utf8',
  )

  for (const label of [
    'First-party',
    'Managed cloud',
    'Gateway',
    'Self-hosted',
  ]) {
    assert.match(page, new RegExp(`['"]${label}['"]`))
  }
  assert.match(page, /relationshipLabel\[binding\.relationship\]/)
})

test('model hub resolves creator marks only from catalog presentation', () => {
  const page = readFileSync(
    resolve(repositoryRoot, 'website/src/components/model-hub/ModelHubMark.tsx'),
    'utf8',
  )

  assert.doesNotMatch(page, /publisherIcons/)
  assert.doesNotMatch(page, /publisher=\{/)
  assert.match(page, /const Icon = packageIcons\[packageID\]/)
  assert.match(page, /directLogo && !logoFailed/)
  assert.match(page, /onError=\{\(\) => setLogoFailed\(true\)\}/)
  assert.match(page, /presentation\.monogram/)
})

test('catalog public logos exist in both application public roots', () => {
  const publicLogos = [...catalog.models, ...catalog.providers]
    .map(entry => entry.presentation?.logo)
    .filter(logo => logo?.startsWith('public:/'))

  assert.ok(publicLogos.length > 0)
  for (const logo of new Set(publicLogos)) {
    const relativePath = logo.slice('public:/'.length)
    assert.ok(
      existsSync(resolve(repositoryRoot, 'website/static', relativePath)),
      `${logo} is missing from the website public root`,
    )
    assert.ok(
      existsSync(
        resolve(repositoryRoot, 'dashboard/frontend/public', relativePath),
      ),
      `${logo} is missing from the Dashboard public root`,
    )
  }
})
