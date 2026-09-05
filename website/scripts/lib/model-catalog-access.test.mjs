import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
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
    resolve(repositoryRoot, 'website/src/pages/models.tsx'),
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
