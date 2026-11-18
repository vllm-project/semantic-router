import { describe, expect, it } from 'vitest'

import type { RouterConfig } from './dashboardPageTypes'
import {
  backendModelLogo,
  buildAgentSnippets,
  buildRequestSnippets,
  buildStartChecklist,
  listPublicModels,
  normalizeBaseUrl,
  resolveGatewayEndpoint,
} from './startSupport'

const config = {
  listeners: [
    { name: 'mgmt-only', port: 9000, apis: { systemone: {} } },
    { name: 'http-8899', port: 8899 },
  ],
  providers: {
    models: [
      { name: 'qwen3-8b', backend_refs: [{ provider: 'vllm' }] },
      { name: 'gpt-5-mini', backend_refs: [{ provider: 'openai' }] },
    ],
  },
  entrypoints: [
    { model_names: ['vllm-sr/balanced'], recipe: 'default', default: true },
    { model_names: ['vllm-sr/private', 'private'], recipe: 'privacy' },
    { api: 'systemone', model_names: ['vllm-sr/auto'], recipe: 'decide' },
  ],
  recipes: [
    {
      name: 'default',
      description: 'Balanced routing',
      routing: {
        signals: { keywords: [{ name: 'code' }], domains: [{ name: 'law' }] },
        decisions: [{ name: 'a', modelRefs: [{ model: 'qwen3-8b' }, { model: 'gpt-5-mini' }] }],
      },
    },
    {
      name: 'privacy',
      routing: { decisions: [{ name: 'p', modelRefs: [{ model: 'qwen3-8b' }] }] },
    },
    { name: 'decide', routing: { decisions: [] } },
  ],
} as unknown as RouterConfig

describe('resolveGatewayEndpoint', () => {
  it('uses the first listener that serves chat, on the dashboard host', () => {
    const endpoint = resolveGatewayEndpoint({
      listeners: config.listeners,
      hostname: 'router.internal',
    })
    expect(endpoint).toMatchObject({
      origin: 'http://router.internal:8899',
      apiBase: 'http://router.internal:8899/v1',
      listener: 'http-8899',
      requiresKey: false,
      source: 'listener',
    })
  })

  it('switches to https for TLS listeners and reports client keys', () => {
    const endpoint = resolveGatewayEndpoint({
      listeners: [{ name: 'tls', port: 8443, tls: { cert_file: 'a' }, api_keys: ['***'] }],
      hostname: '0.0.0.0',
    })
    expect(endpoint.origin).toBe('https://localhost:8443')
    expect(endpoint.requiresKey).toBe(true)
  })

  it('falls back to the setup port and lets a typed base URL win', () => {
    expect(resolveGatewayEndpoint({ fallbackPort: 8801, hostname: 'localhost' }).origin).toBe(
      'http://localhost:8801',
    )
    expect(resolveGatewayEndpoint({ fallbackPort: 0, hostname: 'localhost' }).origin).toBe(
      'http://localhost:8899',
    )
    const custom = resolveGatewayEndpoint({
      listeners: config.listeners,
      hostname: 'localhost',
      override: 'https://gateway.example.com/v1/',
    })
    expect(custom).toMatchObject({
      origin: 'https://gateway.example.com',
      apiBase: 'https://gateway.example.com/v1',
      source: 'custom',
    })
  })

  it('prefers the declared public address to the listener guess, and a typed one to both', () => {
    const declared = {
      listeners: config.listeners,
      hostname: 'dashboard.vllm-sr.ai',
      publicBaseUrl: 'https://mom.vllm-sr.ai/',
      publicRequiresKey: true,
    }
    expect(resolveGatewayEndpoint(declared)).toMatchObject({
      origin: 'https://mom.vllm-sr.ai',
      apiBase: 'https://mom.vllm-sr.ai/v1',
      listener: 'http-8899',
      requiresKey: true,
      source: 'public',
    })
    expect(resolveGatewayEndpoint({ ...declared, override: 'http://10.0.0.5:8899' })).toMatchObject(
      { origin: 'http://10.0.0.5:8899', requiresKey: true, source: 'custom' },
    )
    expect(resolveGatewayEndpoint({ ...declared, publicBaseUrl: 'not a url' })).toMatchObject({
      origin: 'http://dashboard.vllm-sr.ai:8899',
      requiresKey: false,
      source: 'listener',
    })
  })

  it('asks for a key at a public address when the listener behind it checks keys', () => {
    const endpoint = resolveGatewayEndpoint({
      listeners: [{ name: 'keyed', port: 8899, api_key_files: ['/keys'] }],
      hostname: 'localhost',
      publicBaseUrl: 'https://mom.vllm-sr.ai',
    })
    expect(endpoint).toMatchObject({ source: 'public', requiresKey: true })
    expect(
      resolveGatewayEndpoint({
        listeners: config.listeners,
        hostname: 'localhost',
        publicBaseUrl: 'https://mom.vllm-sr.ai',
      }).requiresKey,
    ).toBe(false)
  })

  it('accepts only http(s) base URLs', () => {
    expect(normalizeBaseUrl('ftp://x')).toBeUndefined()
    expect(normalizeBaseUrl('not a url')).toBeUndefined()
    expect(normalizeBaseUrl(' http://a.example:8899/prefix/ ')).toBe('http://a.example:8899/prefix')
  })
})

describe('listPublicModels', () => {
  it('lists chat entrypoints with their recipe facts, then System One entrypoints', () => {
    const models = listPublicModels(config)
    expect(models.map((model) => [model.name, model.api, model.recipe, model.isDefault])).toEqual([
      ['vllm-sr/balanced', 'chat', 'default', true],
      ['vllm-sr/private', 'chat', 'privacy', false],
      ['private', 'chat', 'privacy', false],
      ['vllm-sr/auto', 'systemone', 'decide', false],
    ])
    expect(models[0]).toMatchObject({
      description: 'Balanced routing',
      decisions: 1,
      signals: 2,
      providers: ['vllm', 'openai'],
    })
  })

  it('adds the implicit vllm-sr/auto default when no entrypoint serves the default recipe', () => {
    const models = listPublicModels({ recipes: [{ name: 'default', routing: {} }] })
    expect(models).toHaveLength(1)
    expect(models[0]).toMatchObject({ name: 'vllm-sr/auto', recipe: 'default', isDefault: true })
  })

  it('shows the default model’s logo for a recipe whose decisions name no model', () => {
    const models = listPublicModels({
      providers: {
        defaults: { model: 'qwen3-8b' },
        models: [{ name: 'qwen3-8b', provider_model_id: 'Qwen/Qwen3-8B' }],
      },
      recipes: [{ name: 'default', routing: { decisions: [{ name: 'any' }] } }],
    } as unknown as RouterConfig)
    expect(models[0].providers).toEqual(['qwen'])
  })
})

describe('backendModelLogo', () => {
  it('prefers the card’s logo, then the publishing organisation, then the serving provider', () => {
    const card = {
      name: 'm',
      presentation: { logo: 'package:qwen', monogram: 'Q', monochrome: false },
    }
    expect(
      backendModelLogo('m', {
        name: 'm',
        endpoints: [],
        card_override: card,
        backend_refs: [{ name: 'b', provider: 'vllm' }],
      }),
    ).toBe('qwen')
    expect(
      backendModelLogo('m', { name: 'm', endpoints: [], provider_model_id: 'meta-llama/Llama-3' }),
    ).toBe('meta')
    expect(backendModelLogo('deepseek/deepseek-v4', undefined)).toBe('deepseek')
    expect(
      backendModelLogo('m', {
        name: 'm',
        endpoints: [],
        backend_refs: [{ name: 'b', provider: 'OpenAI' }],
      }),
    ).toBe('openai')
    expect(backendModelLogo('local-model', undefined)).toBe('vllm')
  })
})

describe('snippets', () => {
  const target = {
    origin: 'http://localhost:8899',
    apiBase: 'http://localhost:8899/v1',
    model: 'vllm-sr/balanced',
    requiresKey: false,
  }

  it('builds curl and SDK requests for the chosen model', () => {
    const [curl, python, typescript] = buildRequestSnippets(target)
    expect(curl.code).toContain('curl -sS -i http://localhost:8899/v1/chat/completions')
    expect(curl.code).toContain('"model": "vllm-sr/balanced"')
    expect(curl.code).not.toContain('Authorization')
    expect(python.code).toContain('base_url="http://localhost:8899/v1"')
    expect(python.code).toContain('with_raw_response')
    expect(typescript.code).toContain('.withResponse()')

    const keyed = buildRequestSnippets({ ...target, requiresKey: true })
    expect(keyed[0].code).toContain('Authorization: Bearer $VLLM_SR_API_KEY')
    expect(keyed[1].code).toContain('os.environ["VLLM_SR_API_KEY"]')
  })

  it('takes the agent commands from the agent-harness guide', () => {
    const [claude, codex, opencode, dsh] = buildAgentSnippets(target)
    expect(claude.code).toContain('vllm-sr connect claude-code')
    expect(claude.code).toContain('--base-url http://localhost:8899')
    expect(claude.code).toContain('--model vllm-sr/balanced')
    expect(codex.code).toContain('codex --profile vllm-sr')
    expect(JSON.parse(opencode.code)).toMatchObject({
      provider: {
        'vllm-sr': {
          npm: '@ai-sdk/openai-compatible',
          options: { baseURL: 'http://localhost:8899/v1' },
          models: { balanced: { id: 'vllm-sr/balanced' } },
        },
      },
      model: 'vllm-sr/balanced',
    })
    expect(dsh.code).toContain('baseURL: http://localhost:8899/v1')
    expect(dsh.code).toContain('model: vllm-sr/balanced')

    const auto = buildAgentSnippets({ ...target, model: 'vllm-sr/auto', requiresKey: true })
    expect(auto[0].code).not.toContain('--model')
    expect(auto[0].code).toContain("export VLLM_SR_API_KEY=<the listener's API key>")
  })
})

describe('buildStartChecklist', () => {
  it('reads the live state of each first-run step', () => {
    const steps = buildStartChecklist({
      status: {
        overall: 'healthy',
        version: 'v0.5.0',
        deployment_type: 'docker',
        services: [],
        models: {
          models: [
            {
              name: 'decision',
              type: 'decision',
              loaded: true,
              state: 'ready',
              metadata: { provider: 'model_runtime', deployment: 'primary' },
            },
          ],
        },
      },
      statusFailed: false,
      config,
      replay: { state: 'ready', count: 3, latestAt: new Date(1_000).toISOString() },
      now: 61_000,
    })
    expect(steps.map((step) => [step.id, step.state])).toEqual([
      ['router', 'done'],
      ['backend', 'done'],
      ['decision-model', 'done'],
      ['first-request', 'done'],
    ])
    expect(steps[0].detail).toBe('Healthy · v0.5.0')
    expect(steps[3].detail).toBe('Last request 1m ago')
  })

  it('marks what is missing and what is still loading', () => {
    const steps = buildStartChecklist({
      status: { overall: 'not_running', deployment_type: 'none', services: [] },
      statusFailed: false,
      config: { providers: { models: [] } },
      replay: { state: 'unavailable', count: 0, reason: 'replay_off' },
    })
    expect(steps.map((step) => step.state)).toEqual(['todo', 'todo', 'todo', 'todo'])
    expect(steps[3].detail).toBe('Router Replay is not recording')

    const loading = buildStartChecklist({
      status: null,
      statusFailed: false,
      config: null,
      replay: { state: 'loading', count: 0 },
    })
    expect(loading.map((step) => step.state)).toEqual(['pending', 'pending', 'pending', 'pending'])
  })

  it('reads the decision model step from the state Models reads when the page has it', () => {
    const base = {
      status: { overall: 'healthy', deployment_type: 'kubernetes', services: [] },
      statusFailed: false,
      config: { providers: { models: [] } },
      replay: { state: 'ready' as const, count: 0 },
    }
    const ready = buildStartChecklist({
      ...base,
      decisionModel: { model: 'vela3-turbo', label: 'Ready' },
    })
    expect(ready[2]).toMatchObject({ state: 'done', detail: 'vela3-turbo is ready' })
    const degraded = buildStartChecklist({
      ...base,
      decisionModel: { model: 'vela3-turbo', label: 'Degraded capacity' },
    })
    expect(degraded[2]).toMatchObject({ state: 'todo', detail: 'vela3-turbo · Degraded capacity' })
  })
})
