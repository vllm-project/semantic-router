import assert from 'node:assert/strict'
import { existsSync, readdirSync, readFileSync } from 'node:fs'
import { join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { test } from 'node:test'
import { load } from 'js-yaml'
import recipeDiscoveryPlugin from '../../src/plugins/recipeDiscovery.cjs'

const siteDir = fileURLToPath(new URL('../../', import.meta.url))
const recipeRoot = resolve(siteDir, '../config/recipes')
const catalog = recipeDiscoveryPlugin({ siteDir, baseUrl: '/' }).loadContent()

test('discovers maintained standalone recipes and publishes the exact source YAML', () => {
  const expected = readdirSync(recipeRoot)
    .filter(name => existsSync(join(recipeRoot, name, 'metadata.yaml')))
    .sort()
  assert.deepEqual(catalog.recipes.map(recipe => recipe.id), expected)
  assert.ok(!expected.includes('built-in'))

  for (const recipe of catalog.recipes) {
    const source = readFileSync(resolve(siteDir, '..', recipe.configPath), 'utf8')
    const config = load(source)
    const metadata = load(readFileSync(join(recipeRoot, recipe.id, 'metadata.yaml'), 'utf8'))
    assert.equal(recipe.yaml, source, recipe.id)
    assert.equal(recipe.name, metadata.name)
    assert.deepEqual(recipe.tags, metadata.tags ?? [])
    assert.deepEqual(recipe.entrypoints, config.entrypoints?.length
      ? config.entrypoints.flatMap(entrypoint => entrypoint.model_names)
      : ['vllm-sr/auto'])
    assert.ok(recipe.limitations.length > 0, recipe.id)
    assert.deepEqual(recipe.providers.map(provider => provider.name), config.providers.models.map(provider => provider.name))
  }
})

test('projects model evidence and explicit AMD requirements without inferring support', () => {
  const agent = catalog.recipes.find(recipe => recipe.id === 'agent')
  assert.ok(agent.tags.includes('agents'))
  const decision = catalog.decisionModels.find(model => model.artifact === agent.decision.artifact)
  assert.ok(decision.examples.some(example => example.name === agent.name && example.tasks.length > 0))
  assert.equal(new Set(catalog.decisionModels.map(model => model.artifact)).size, catalog.decisionModels.length)

  const amd = catalog.recipes.find(recipe => recipe.id === 'vela-amd')
  assert.deepEqual(amd.entrypoints, ['vela-auto'])
  assert.match(amd.decision.device, /^rocm/)
  assert.match(amd.embedding.device, /^rocm/)
  assert.equal(amd.servePlatform, 'rocm')
})

test('registers the localized route and keeps recipe YAML in route data', async () => {
  const plugin = recipeDiscoveryPlugin({ siteDir, baseUrl: '/zh-Hans/' })
  const routes = []
  await plugin.contentLoaded({
    content: catalog,
    actions: {
      createData(name, content) {
        assert.equal(name, 'recipes.json')
        assert.equal(JSON.parse(content).recipes[0].yaml, catalog.recipes[0].yaml)
        return `/generated/${name}`
      },
      addRoute(route) { routes.push(route) },
    },
  })
  assert.equal(routes.length, 1)
  assert.equal(routes[0].path, '/zh-Hans/recipes')
  assert.deepEqual(routes[0].modules, { catalog: '/generated/recipes.json' })
})
