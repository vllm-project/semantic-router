export type ModelProviderCategory = 'Local runtimes' | 'Model APIs' | 'Private gateways'

export interface ModelProviderDefinition {
  id: string
  providerCode: string
  name: string
  shortName: string
  description: string
  category: ModelProviderCategory
  accent: string
  baseUrl: string
  apiFormat: 'openai' | 'anthropic'
  runtimeProvider: 'openai' | 'anthropic'
  authHeader: string
  authPrefix: string
  chatPath?: string
  modelsPath?: string
  extraHeaders?: Record<string, string>
  apiKeyOptional?: boolean
}

type ProviderOverrides = Partial<
  Omit<
    ModelProviderDefinition,
    'id' | 'providerCode' | 'name' | 'shortName' | 'description' | 'category' | 'accent'
  >
>

const hostedEndpoint = (host: string, path = ''): string => `https://${host}${path}`

const local = (
  id: string,
  providerCode: string,
  name: string,
  shortName: string,
  description: string,
  accent: string,
  overrides: ProviderOverrides = {},
): ModelProviderDefinition => ({
  id,
  providerCode,
  name,
  shortName,
  description,
  category: 'Local runtimes',
  accent,
  baseUrl: '',
  apiFormat: 'openai',
  runtimeProvider: 'openai',
  authHeader: 'Authorization',
  authPrefix: 'Bearer',
  apiKeyOptional: true,
  ...overrides,
})

const api = (
  id: string,
  providerCode: string,
  name: string,
  shortName: string,
  description: string,
  accent: string,
  baseUrl: string,
  overrides: ProviderOverrides = {},
): ModelProviderDefinition => ({
  id,
  providerCode,
  name,
  shortName,
  description,
  category: 'Model APIs',
  accent,
  baseUrl,
  modelsPath: '/models',
  apiFormat: 'openai',
  runtimeProvider: 'openai',
  authHeader: 'Authorization',
  authPrefix: 'Bearer',
  ...overrides,
})

const gateway = (
  id: string,
  providerCode: string,
  name: string,
  shortName: string,
  description: string,
  accent: string,
  overrides: ProviderOverrides = {},
): ModelProviderDefinition => ({
  ...api(id, providerCode, name, shortName, description, accent, '', overrides),
  category: 'Private gateways',
  modelsPath: undefined,
})

export const MODEL_PROVIDERS: readonly ModelProviderDefinition[] = [
  local('vllm', 'VLLM', 'vLLM', 'V', 'Serve any OpenAI-compatible vLLM endpoint.', '#5ba8ff'),
  local(
    'sglang',
    'CUSTOM_OPENAI',
    'SGLang',
    'S',
    'Connect an SGLang OpenAI API server.',
    '#d55816',
  ),
  local(
    'amd-atom',
    'CUSTOM_OPENAI',
    'AMD ATOM',
    'A',
    'Connect an AMD ATOM inference endpoint.',
    '#ed1c24',
  ),
  local(
    'openai-compatible',
    'OpenAI_Compatible',
    'OpenAI compatible',
    'OA',
    'Connect any compatible /v1 endpoint.',
    '#9ca3af',
  ),

  api(
    'openrouter',
    'Openrouter',
    'OpenRouter',
    'OR',
    'One key for models across providers.',
    '#7c5cff',
    hostedEndpoint('openrouter.ai', '/api/v1'),
  ),
  api(
    'openai',
    'OpenAI',
    'OpenAI',
    '◌',
    'GPT and o-series models from OpenAI.',
    '#10a37f',
    hostedEndpoint('api.openai.com', '/v1'),
  ),
  api(
    'anthropic',
    'Anthropic',
    'Anthropic',
    'AI',
    'Claude through the native Messages API.',
    '#d97757',
    hostedEndpoint('api.anthropic.com'),
    {
      apiFormat: 'anthropic',
      runtimeProvider: 'anthropic',
      authHeader: 'x-api-key',
      authPrefix: '',
      chatPath: '/v1/messages',
      modelsPath: '/v1/models',
      extraHeaders: { 'anthropic-version': '2023-06-01' },
    },
  ),
  api(
    'google-ai-studio',
    'Google_AI_Studio',
    'Google AI Studio',
    'G',
    'Gemini through its OpenAI-compatible API.',
    '#4285f4',
    hostedEndpoint('generativelanguage.googleapis.com', '/v1beta/openai'),
  ),
  api(
    'deepseek',
    'Deepseek',
    'DeepSeek',
    'DS',
    'DeepSeek chat and reasoning models.',
    '#4d6bfe',
    hostedEndpoint('api.deepseek.com', '/v1'),
  ),
  api(
    'mistral',
    'MistralAI',
    'Mistral AI',
    'M',
    'Mistral and Codestral APIs.',
    '#ff7000',
    hostedEndpoint('api.mistral.ai', '/v1'),
  ),
  api(
    'groq',
    'Groq',
    'Groq',
    'GQ',
    'Low-latency inference on GroqCloud.',
    '#f55036',
    hostedEndpoint('api.groq.com', '/openai/v1'),
  ),
  api(
    'together-ai',
    'TogetherAI',
    'Together AI',
    'TO',
    'Open models on Together AI.',
    '#14b8a6',
    hostedEndpoint('api.together.xyz', '/v1'),
  ),
  api(
    'fireworks-ai',
    'FireworksAI',
    'Fireworks AI',
    'FW',
    'Serverless and dedicated inference.',
    '#f97316',
    hostedEndpoint('api.fireworks.ai', '/inference/v1'),
  ),
  api(
    'cerebras',
    'Cerebras',
    'Cerebras',
    'C',
    'Fast inference on Cerebras systems.',
    '#facc15',
    hostedEndpoint('api.cerebras.ai', '/v1'),
  ),
  api(
    'xai',
    'xAI',
    'xAI',
    'x',
    'Grok models from xAI.',
    '#f4f4f5',
    hostedEndpoint('api.x.ai', '/v1'),
  ),
  api(
    'perplexity',
    'Perplexity',
    'Perplexity',
    'P',
    'Online models with search grounding.',
    '#20b8a6',
    hostedEndpoint('api.perplexity.ai'),
    { modelsPath: '/v1/models' },
  ),
  api(
    'cohere',
    'COHERE_CHAT',
    'Cohere',
    'CO',
    'Command models through OpenAI compatibility.',
    '#39594d',
    hostedEndpoint('api.cohere.ai', '/compatibility/v1'),
  ),
  api(
    'deepinfra',
    'DeepInfra',
    'DeepInfra',
    'DI',
    'Serverless open-model inference.',
    '#8b5cf6',
    hostedEndpoint('api.deepinfra.com', '/v1/openai'),
  ),
  api(
    'hugging-face',
    'HUGGINGFACE',
    'Hugging Face',
    'HF',
    'Inference Providers behind one API.',
    '#ffcc4d',
    hostedEndpoint('router.huggingface.co', '/v1'),
  ),
  api(
    'nvidia-nim',
    'NVIDIA_NIM',
    'NVIDIA NIM',
    'N',
    'Hosted NIM model endpoints.',
    '#8aae42',
    hostedEndpoint('integrate.api.nvidia.com', '/v1'),
  ),
  api(
    'sambanova',
    'Sambanova',
    'SambaNova',
    'SN',
    'SambaNova Cloud model APIs.',
    '#f43f5e',
    hostedEndpoint('api.sambanova.ai', '/v1'),
  ),
  api(
    'dashscope',
    'Dashscope',
    'DashScope',
    'Q',
    'Qwen models through Alibaba Cloud.',
    '#615ced',
    hostedEndpoint('dashscope.aliyuncs.com', '/compatible-mode/v1'),
  ),
  api(
    'minimax',
    'MINIMAX',
    'MiniMax',
    'MM',
    'MiniMax text and reasoning models.',
    '#ff4d6d',
    hostedEndpoint('api.minimax.io', '/v1'),
  ),
  api(
    'moonshot',
    'MOONSHOT',
    'Moonshot AI',
    'K',
    'Kimi models through Moonshot AI.',
    '#f4f4f5',
    hostedEndpoint('api.moonshot.ai', '/v1'),
  ),
  api(
    'zai',
    'ZAI',
    'Z.AI',
    'Z',
    'GLM models from Z.AI.',
    '#2563eb',
    hostedEndpoint('api.z.ai', '/api/paas/v4'),
  ),
  api(
    'novita',
    'NOVITA',
    'Novita AI',
    'NV',
    'Serverless open-model inference.',
    '#7c3aed',
    hostedEndpoint('api.novita.ai', '/v3/openai'),
  ),
  api(
    'nebius',
    'NEBIUS',
    'Nebius AI Studio',
    'NB',
    'Open models from Nebius AI Studio.',
    '#8b5cf6',
    hostedEndpoint('api.studio.nebius.com', '/v1'),
  ),
  api(
    'featherless',
    'FEATHERLESS_AI',
    'Featherless AI',
    'FL',
    'On-demand open-model inference.',
    '#d946ef',
    hostedEndpoint('api.featherless.ai', '/v1'),
  ),
  api(
    'friendli',
    'FRIENDLIAI',
    'FriendliAI',
    'FR',
    'Optimized serverless model endpoints.',
    '#ff4f64',
    hostedEndpoint('api.friendli.ai', '/serverless/v1'),
  ),
  api(
    'vercel-ai-gateway',
    'VERCEL_AI_GATEWAY',
    'Vercel AI Gateway',
    '▲',
    'Models through Vercel AI Gateway.',
    '#f4f4f5',
    hostedEndpoint('ai-gateway.vercel.sh', '/v1'),
  ),
  api(
    'cometapi',
    'COMETAPI',
    'CometAPI',
    'CA',
    'Unified access to hosted models.',
    '#38bdf8',
    hostedEndpoint('api.cometapi.com', '/v1'),
  ),
  api(
    'sakana',
    'CUSTOM_OPENAI',
    'Sakana AI',
    'SA',
    'Models from Sakana AI.',
    '#e10600',
    hostedEndpoint('api.sakana.ai', '/v1'),
  ),

  gateway(
    'ollama',
    'OLLAMA_CHAT',
    'Ollama',
    'O',
    'Connect a local or remote Ollama server.',
    '#f4f4f5',
    { apiKeyOptional: true },
  ),
  gateway(
    'lm-studio',
    'LM_STUDIO',
    'LM Studio',
    'LM',
    'Connect an LM Studio inference server.',
    '#5b8cff',
    { apiKeyOptional: true },
  ),
  gateway(
    'xinference',
    'XINFERENCE',
    'Xinference',
    'XI',
    'Connect a Xinference deployment.',
    '#5b8cff',
    { apiKeyOptional: true },
  ),
  gateway(
    'docker-model-runner',
    'DOCKER_MODEL_RUNNER',
    'Docker Model Runner',
    'D',
    'Connect Docker Model Runner.',
    '#2496ed',
    { apiKeyOptional: true },
  ),
  gateway('lemonade', 'LEMONADE', 'Lemonade', 'L', 'Connect a local Lemonade server.', '#facc15', {
    apiKeyOptional: true,
  }),
  gateway(
    'nvidia-riva',
    'NVIDIA_RIVA',
    'NVIDIA Riva',
    'NR',
    'Connect a private NVIDIA Riva endpoint.',
    '#8aae42',
  ),
  gateway(
    'triton',
    'Triton',
    'NVIDIA Triton',
    'T',
    'Connect an OpenAI-compatible Triton gateway.',
    '#8aae42',
    { apiKeyOptional: true },
  ),
] as const

export const FEATURED_MODEL_PROVIDERS = MODEL_PROVIDERS.slice(0, 4)

export function getModelProvider(providerId?: string): ModelProviderDefinition {
  const normalized = (providerId || '').trim().toLowerCase()
  return (
    MODEL_PROVIDERS.find(
      (provider) =>
        provider.id === normalized || provider.providerCode.toLowerCase() === normalized,
    ) ?? MODEL_PROVIDERS.find((provider) => provider.id === 'openai-compatible')!
  )
}

export function filterModelProviders(query: string): ModelProviderDefinition[] {
  const normalized = query.trim().toLowerCase()
  if (!normalized) return [...MODEL_PROVIDERS]
  return MODEL_PROVIDERS.filter((provider) =>
    [provider.name, provider.id, provider.providerCode, provider.description].some((value) =>
      value.toLowerCase().includes(normalized),
    ),
  )
}
