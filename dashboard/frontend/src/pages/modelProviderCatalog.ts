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
  extraHeaders?: Record<string, string>
  apiKeyOptional?: boolean
}

type ProviderOverrides = Partial<
  Omit<
    ModelProviderDefinition,
    'id' | 'providerCode' | 'name' | 'shortName' | 'description' | 'category' | 'accent'
  >
>

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
  baseUrl = '',
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
})

// OpenAI-compatible providers use the generic OpenAI wire adapter. Anthropic
// uses the native Messages adapter.
export const MODEL_PROVIDERS: readonly ModelProviderDefinition[] = [
  local('vllm', 'VLLM', 'vLLM', 'V', 'Serve any OpenAI-compatible vLLM endpoint.', '#5ba8ff'),
  local(
    'sglang',
    'CUSTOM_OPENAI',
    'SGLang',
    'S',
    'Connect an SGLang OpenAI API server.',
    '#a78bfa',
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
    'https://openrouter.ai/api/v1',
  ),
  api(
    'openai',
    'OpenAI',
    'OpenAI',
    '◌',
    'GPT and o-series models from OpenAI.',
    '#10a37f',
    'https://api.openai.com/v1',
  ),
  api(
    'anthropic',
    'Anthropic',
    'Anthropic',
    'AI',
    'Claude models through the native Messages API.',
    '#d97757',
    'https://api.anthropic.com',
    {
      apiFormat: 'anthropic',
      runtimeProvider: 'anthropic',
      authHeader: 'x-api-key',
      authPrefix: '',
      chatPath: '/v1/messages',
      extraHeaders: { 'anthropic-version': '2023-06-01' },
    },
  ),
  api(
    'google-ai-studio',
    'Google_AI_Studio',
    'Google AI Studio',
    'G',
    'Gemini through Google AI Studio.',
    '#4285f4',
    'https://generativelanguage.googleapis.com/v1beta/openai',
  ),
  api('azure-openai', 'Azure', 'Azure OpenAI', 'AZ', 'Azure-hosted OpenAI deployments.', '#0089d6'),
  api(
    'azure-ai-foundry',
    'Azure_AI_Studio',
    'Azure AI Foundry',
    'AF',
    'Models deployed from Azure AI Foundry.',
    '#31a8ff',
  ),
  api(
    'amazon-bedrock',
    'Bedrock',
    'Amazon Bedrock',
    'AWS',
    'Foundation models served by Amazon Bedrock.',
    '#ff9900',
  ),
  api(
    'vertex-ai',
    'Vertex_AI',
    'Vertex AI',
    'VX',
    'Gemini and partner models on Vertex AI.',
    '#4285f4',
  ),
  api(
    'deepseek',
    'Deepseek',
    'DeepSeek',
    'DS',
    'DeepSeek chat and reasoning models.',
    '#4d6bfe',
    'https://api.deepseek.com',
  ),
  api(
    'mistral',
    'MistralAI',
    'Mistral AI',
    'M',
    'Mistral and Codestral APIs.',
    '#ff7000',
    'https://api.mistral.ai/v1',
  ),
  api(
    'groq',
    'Groq',
    'Groq',
    'GQ',
    'Low-latency inference on GroqCloud.',
    '#f55036',
    'https://api.groq.com/openai/v1',
  ),
  api(
    'together-ai',
    'TogetherAI',
    'Together AI',
    'TO',
    'Open models on Together AI.',
    '#14b8a6',
    'https://api.together.xyz/v1',
  ),
  api(
    'fireworks-ai',
    'FireworksAI',
    'Fireworks AI',
    'FW',
    'Fast serverless and dedicated inference.',
    '#f97316',
    'https://api.fireworks.ai/inference/v1',
  ),
  api(
    'cerebras',
    'Cerebras',
    'Cerebras',
    'C',
    'Fast inference on Cerebras systems.',
    '#facc15',
    'https://api.cerebras.ai/v1',
  ),
  api('xai', 'xAI', 'xAI', 'x', 'Grok models from xAI.', '#f4f4f5', 'https://api.x.ai/v1'),
  api(
    'perplexity',
    'Perplexity',
    'Perplexity',
    'P',
    'Online models with search grounding.',
    '#20b8a6',
    'https://api.perplexity.ai',
  ),
  api(
    'cohere',
    'COHERE_CHAT',
    'Cohere',
    'CO',
    'Command models from Cohere.',
    '#39594d',
    'https://api.cohere.ai/compatibility/v1',
  ),
  api('ai21', 'AI21_CHAT', 'AI21', '21', 'Jamba and Jurassic models from AI21.', '#6d5dfc'),
  api(
    'deepinfra',
    'DeepInfra',
    'DeepInfra',
    'DI',
    'Serverless open-model inference.',
    '#8b5cf6',
    'https://api.deepinfra.com/v1/openai',
  ),
  api(
    'hugging-face',
    'HUGGINGFACE',
    'Hugging Face',
    'HF',
    'Inference endpoints and providers.',
    '#ffcc4d',
  ),
  api('replicate', 'REPLICATE', 'Replicate', 'R', 'Hosted open-source models.', '#f4f4f5'),
  api(
    'databricks',
    'Databricks',
    'Databricks',
    'DB',
    'Foundation Model APIs on Databricks.',
    '#ff3621',
  ),
  api(
    'cloudflare',
    'CLOUDFLARE',
    'Cloudflare Workers AI',
    'CF',
    'Models on the Cloudflare network.',
    '#f38020',
  ),
  api(
    'nvidia-nim',
    'NVIDIA_NIM',
    'NVIDIA NIM',
    'N',
    'NVIDIA-hosted and private NIM endpoints.',
    '#8aae42',
    'https://integrate.api.nvidia.com/v1',
  ),
  api(
    'sambanova',
    'Sambanova',
    'SambaNova',
    'SN',
    'SambaNova Cloud model APIs.',
    '#f43f5e',
    'https://api.sambanova.ai/v1',
  ),
  api(
    'snowflake',
    'Snowflake',
    'Snowflake Cortex',
    '❄',
    'Models available through Snowflake Cortex.',
    '#29b5e8',
  ),
  api(
    'watsonx',
    'WATSONX',
    'IBM watsonx',
    'IBM',
    'Enterprise foundation models on watsonx.',
    '#0f62fe',
  ),
  api(
    'oracle-oci',
    'Oracle',
    'Oracle OCI',
    'OCI',
    'Generative AI hosted on Oracle Cloud.',
    '#c74634',
  ),
  api(
    'sap',
    'SAP',
    'SAP Generative AI Hub',
    'SAP',
    'Enterprise models through SAP AI Core.',
    '#0a6ed1',
  ),
  api('volcengine', 'VolcEngine', 'VolcEngine', 'VE', 'Model APIs from VolcEngine.', '#3370ff'),
  api(
    'dashscope',
    'Dashscope',
    'DashScope',
    'Q',
    'Qwen models through Alibaba DashScope.',
    '#615ced',
  ),
  api('minimax', 'MINIMAX', 'MiniMax', 'MM', 'MiniMax text and reasoning models.', '#ff4d6d'),
  api('moonshot', 'MOONSHOT', 'Moonshot', 'K', 'Kimi models through Moonshot AI.', '#f4f4f5'),
  api('zai', 'ZAI', 'Z.AI', 'Z', 'GLM models from Z.AI.', '#2563eb'),
  api(
    'novita',
    'NOVITA',
    'Novita AI',
    'NV',
    'Serverless open-model inference.',
    '#7c3aed',
    'https://api.novita.ai/v3/openai',
  ),
  api(
    'baseten',
    'BASETEN',
    'Baseten',
    'B',
    'Dedicated and serverless model deployments.',
    '#ff6b35',
  ),
  api(
    'nebius',
    'NEBIUS',
    'Nebius AI Studio',
    'NB',
    'Open models from Nebius AI Studio.',
    '#8b5cf6',
    'https://api.studio.nebius.com/v1',
  ),
  api(
    'hyperbolic',
    'HYPERBOLIC',
    'Hyperbolic',
    'HY',
    'Open-model inference on Hyperbolic.',
    '#7c3aed',
    'https://api.hyperbolic.xyz/v1',
  ),
  api(
    'featherless',
    'FEATHERLESS_AI',
    'Featherless AI',
    'FL',
    'On-demand open-model inference.',
    '#d946ef',
  ),
  api(
    'friendli',
    'FRIENDLIAI',
    'FriendliAI',
    'FR',
    'Optimized model endpoints from FriendliAI.',
    '#ff4f64',
  ),
  api('lambda', 'LAMBDA_AI', 'Lambda', 'λ', 'Model inference on Lambda Cloud.', '#6bff81'),
  api(
    'vercel-ai-gateway',
    'VERCEL_AI_GATEWAY',
    'Vercel AI Gateway',
    '▲',
    'Models through Vercel AI Gateway.',
    '#f4f4f5',
  ),
  api('cometapi', 'COMETAPI', 'CometAPI', 'CA', 'Unified model API from CometAPI.', '#38bdf8'),
  api('aiml-api', 'AIML', 'AI/ML API', 'ML', 'Unified API for hosted AI models.', '#22c55e'),
  api('bytez', 'BYTEZ', 'Bytez', 'BY', 'Hosted model APIs from Bytez.', '#06b6d4'),
  api('nscale', 'NSCALE', 'Nscale', 'NS', 'Inference services from Nscale.', '#10b981'),
  api('ovhcloud', 'OVHCLOUD', 'OVHcloud', 'OVH', 'AI endpoints on OVHcloud.', '#0050d7'),
  api('heroku', 'HEROKU', 'Heroku AI', 'H', 'Managed model inference on Heroku.', '#79589f'),
  api('galadriel', 'GALADRIEL', 'Galadriel', 'GA', 'Open model APIs from Galadriel.', '#a855f7'),
  api('empower', 'EMPOWER', 'Empower', 'E', 'Enterprise model endpoints.', '#3b82f6'),
  api('predibase', 'PREDIBASE', 'Predibase', 'PB', 'Fine-tuned and open model serving.', '#7c3aed'),
  api('maritalk', 'MARITALK', 'Maritalk', 'MA', 'Portuguese-first model APIs.', '#16a34a'),
  api('nlp-cloud', 'NLP_CLOUD', 'NLP Cloud', 'NC', 'Production NLP and LLM APIs.', '#0ea5e9'),
  api('clarifai', 'CLARIFAI', 'Clarifai', 'CL', 'Models and workflows on Clarifai.', '#8b5cf6'),
  api(
    'github-models',
    'GITHUB',
    'GitHub Models',
    'GH',
    'Model APIs provided by GitHub.',
    '#f4f4f5',
  ),
  api(
    'github-copilot',
    'GITHUB_COPILOT',
    'GitHub Copilot',
    'GC',
    'Copilot model access through GitHub.',
    '#f4f4f5',
  ),
  api('cursor', 'CURSOR', 'Cursor', 'CU', 'Cursor model gateway connections.', '#f4f4f5'),

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
    'llamafile',
    'LLAMAFILE',
    'Llamafile',
    'LF',
    'Connect a llamafile OpenAI endpoint.',
    '#f59e0b',
    { apiKeyOptional: true },
  ),
  gateway(
    'oobabooga',
    'OOBABOOGA',
    'Oobabooga',
    'OO',
    'Connect a text-generation-webui API.',
    '#f97316',
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
