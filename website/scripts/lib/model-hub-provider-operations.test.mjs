import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { test } from 'node:test'
import ts from 'typescript'

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
const helperPath = resolve(
  repositoryRoot,
  'website/src/data/modelHubProviderOperations.ts',
)
const helperJavaScript = ts.transpileModule(readFileSync(helperPath, 'utf8'), {
  compilerOptions: {
    module: ts.ModuleKind.ESNext,
    target: ts.ScriptTarget.ES2020,
  },
  fileName: helperPath,
}).outputText
const support = await import(
  `data:text/javascript;base64,${Buffer.from(helperJavaScript).toString('base64')}`,
)

const protocolFor = provider =>
  catalog.protocols.find(protocol => protocol.id === provider.protocols[0])

test('provider operations mirror the Go registry effective-path contract', () => {
  const fixtures = [
    ['openai', 'openai/responses@1', 'create', '/v1/responses'],
    ['openrouter', 'openai/chat-completions@1', 'create', '/api/v1/chat/completions'],
    ['openrouter', 'openai/responses@1', 'create', '/api/v1/responses'],
    ['gemini', 'openai/chat-completions@1', 'create', '/v1beta/openai/chat/completions'],
    ['gemini', 'openai/chat-completions@1', 'list_models', '/v1beta/openai/models'],
    [
      'baidu-ai-studio',
      'openai/chat-completions@1',
      'create',
      '/llm/lmapi/v3/chat/completions',
    ],
    ['bedrock', 'openai/chat-completions@1', 'create', '/chat/completions'],
    ['minimax', 'openai/chat-completions@1', 'create', '/v1/chat/completions'],
  ]

  for (const [providerID, protocolID, operationID, expectedPath] of fixtures) {
    const provider = catalog.providers.find(candidate => candidate.id === providerID)
    const protocol = catalog.protocols.find(candidate => candidate.id === protocolID)
    assert.ok(provider, providerID)
    assert.ok(protocol, protocolID)
    const operation = support
      .providerProtocolOperations(provider, protocol)
      .find(candidate => candidate.id === operationID)

    assert.equal(operation?.path, expectedPath, `${providerID} ${protocolID}#${operationID}`)
  }

  for (const providerID of ['bedrock', 'minimax']) {
    const provider = catalog.providers.find(candidate => candidate.id === providerID)
    assert.deepEqual(
      support.providerProtocolOperations(provider, protocolFor(provider)).map(({ id }) => id),
      ['create'],
    )
  }
})

test('provider cards render each supported operation instead of only a count', () => {
  const page = readFileSync(
    resolve(repositoryRoot, 'website/src/components/model-hub/ModelHubProviders.tsx'),
    'utf8',
  )
  const providerCard = page.slice(
    page.indexOf('function ProviderCard'),
    page.indexOf('export function ModelHubProviders'),
  )

  assert.match(
    providerCard,
    /providerProtocolOperations\(provider, protocol\)/,
  )
  assert.match(providerCard, /operation\.method/)
  assert.match(providerCard, /readable\(operation\.id\)/)
  assert.match(providerCard, /operation\.id/)
  assert.match(providerCard, /operation\.path/)
  assert.doesNotMatch(providerCard, /operationCount/)
})
