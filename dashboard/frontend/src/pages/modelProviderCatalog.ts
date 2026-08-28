import { modelProviderIconAssets, monochromeModelProviderIcons } from './modelProviderIcons'

export type ProviderAuthMode = 'none' | 'bearer' | 'anthropic'

export interface ModelProviderPreset {
  id: string
  name: string
  description: string
  category: 'Start here' | 'Model APIs' | 'Private runtimes'
  baseUrl: string
  apiFormat: string
  authMode: ProviderAuthMode
  icon: string
  monogram: string
}

const lobeIcon = (name: string) => modelProviderIconAssets[name] ?? ''

const lobeMonoIcon = lobeIcon

export const isMonochromeModelProviderIcon = (icon: string): boolean =>
  monochromeModelProviderIcons.has(icon)

export const modelProviderCatalog: ModelProviderPreset[] = [
  {
    id: 'vllm',
    name: 'vLLM',
    description: 'Connect a private vLLM endpoint.',
    category: 'Start here',
    baseUrl: '',
    apiFormat: 'openai',
    authMode: 'bearer',
    icon: lobeIcon('vllm'),
    monogram: 'v',
  },
  {
    id: 'sglang',
    name: 'SGLang',
    description: 'Connect a private SGLang endpoint.',
    category: 'Start here',
    baseUrl: '',
    apiFormat: 'openai',
    authMode: 'bearer',
    icon: 'https://raw.githubusercontent.com/sgl-project/sgl-docs/main/favicon.png',
    monogram: 'S',
  },
  {
    id: 'amd-atom',
    name: 'AMD ATOM',
    description: 'Connect an AMD ATOM deployment.',
    category: 'Start here',
    baseUrl: '',
    apiFormat: 'openai',
    authMode: 'bearer',
    icon: '/amd.png',
    monogram: 'A',
  },
  {
    id: 'openai-compatible',
    name: 'OpenAI Compatible',
    description: 'Connect any compatible private endpoint.',
    category: 'Start here',
    baseUrl: '',
    apiFormat: 'openai',
    authMode: 'bearer',
    icon: lobeIcon('openai'),
    monogram: 'O',
  },
  {
    id: 'openrouter',
    name: 'OpenRouter',
    description: 'One API for a broad model catalog.',
    category: 'Model APIs',
    baseUrl: 'https://openrouter.ai/api/v1',
    apiFormat: 'openai',
    authMode: 'bearer',
    icon: lobeIcon('openrouter'),
    monogram: 'OR',
  },
  {
    id: 'openai',
    name: 'OpenAI',
    description: 'OpenAI models and reasoning families.',
    category: 'Model APIs',
    baseUrl: 'https://api.openai.com/v1',
    apiFormat: 'openai',
    authMode: 'bearer',
    icon: lobeIcon('openai'),
    monogram: 'O',
  },
  {
    id: 'anthropic',
    name: 'Anthropic',
    description: 'Claude models through the native API.',
    category: 'Model APIs',
    baseUrl: 'https://api.anthropic.com/v1',
    apiFormat: 'anthropic',
    authMode: 'anthropic',
    icon: lobeIcon('anthropic'),
    monogram: 'A',
  },
  {
    id: 'anthropic-compatible',
    name: 'Anthropic Compatible',
    description: 'Connect a private Anthropic-compatible endpoint.',
    category: 'Private runtimes',
    baseUrl: '',
    apiFormat: 'anthropic',
    authMode: 'anthropic',
    icon: lobeMonoIcon('anthropic'),
    monogram: 'A',
  },
  {
    id: 'google',
    name: 'Google Gemini',
    description: 'Gemini through its OpenAI-compatible API.',
    category: 'Model APIs',
    baseUrl: 'https://generativelanguage.googleapis.com/v1beta/openai',
    apiFormat: 'openai',
    authMode: 'bearer',
    icon: lobeIcon('gemini'),
    monogram: 'G',
  },
  {
    id: 'deepseek',
    name: 'DeepSeek',
    description: 'DeepSeek chat and reasoning models.',
    category: 'Model APIs',
    baseUrl: 'https://api.deepseek.com/v1',
    apiFormat: 'openai',
    authMode: 'bearer',
    icon: lobeIcon('deepseek'),
    monogram: 'D',
  },
  {
    id: 'groq',
    name: 'Groq',
    description: 'Low-latency hosted inference.',
    category: 'Model APIs',
    baseUrl: 'https://api.groq.com/openai/v1',
    apiFormat: 'openai',
    authMode: 'bearer',
    icon: lobeIcon('groq'),
    monogram: 'G',
  },
  {
    id: 'together',
    name: 'Together AI',
    description: 'Open models through a hosted API.',
    category: 'Model APIs',
    baseUrl: 'https://api.together.ai/v1',
    apiFormat: 'openai',
    authMode: 'bearer',
    icon: lobeIcon('together'),
    monogram: 'T',
  },
  {
    id: 'fireworks',
    name: 'Fireworks AI',
    description: 'Fast serverless model inference.',
    category: 'Model APIs',
    baseUrl: 'https://api.fireworks.ai/inference/v1',
    apiFormat: 'openai',
    authMode: 'bearer',
    icon: lobeIcon('fireworks'),
    monogram: 'F',
  },
  {
    id: 'mistral',
    name: 'Mistral AI',
    description: 'Mistral chat and code models.',
    category: 'Model APIs',
    baseUrl: 'https://api.mistral.ai/v1',
    apiFormat: 'openai',
    authMode: 'bearer',
    icon: lobeIcon('mistral'),
    monogram: 'M',
  },
  {
    id: 'xai',
    name: 'xAI',
    description: 'Grok models through an OpenAI-style API.',
    category: 'Model APIs',
    baseUrl: 'https://api.x.ai/v1',
    apiFormat: 'openai',
    authMode: 'bearer',
    icon: lobeIcon('xai'),
    monogram: 'x',
  },
  {
    id: 'cerebras',
    name: 'Cerebras',
    description: 'High-speed hosted inference.',
    category: 'Model APIs',
    baseUrl: 'https://api.cerebras.ai/v1',
    apiFormat: 'openai',
    authMode: 'bearer',
    icon: lobeIcon('cerebras'),
    monogram: 'C',
  },
  {
    id: 'nvidia-nim',
    name: 'NVIDIA NIM',
    description: 'Hosted models from the NIM catalog.',
    category: 'Model APIs',
    baseUrl: 'https://integrate.api.nvidia.com/v1',
    apiFormat: 'openai',
    authMode: 'bearer',
    icon: lobeIcon('nvidia'),
    monogram: 'N',
  },
  {
    id: 'perplexity',
    name: 'Perplexity',
    description: 'Search-grounded models and online answers.',
    category: 'Model APIs',
    baseUrl: 'https://api.perplexity.ai',
    apiFormat: 'openai',
    authMode: 'bearer',
    icon: lobeIcon('perplexity'),
    monogram: 'P',
  },
  {
    id: 'cohere',
    name: 'Cohere',
    description: 'Command models through the compatibility API.',
    category: 'Model APIs',
    baseUrl: 'https://api.cohere.com/compatibility/v1',
    apiFormat: 'openai',
    authMode: 'bearer',
    icon: lobeIcon('cohere'),
    monogram: 'C',
  },
  {
    id: 'deepinfra',
    name: 'DeepInfra',
    description: 'Serverless inference across open models.',
    category: 'Model APIs',
    baseUrl: 'https://api.deepinfra.com/v1/openai',
    apiFormat: 'openai',
    authMode: 'bearer',
    icon: lobeIcon('deepinfra'),
    monogram: 'D',
  },
  {
    id: 'hugging-face',
    name: 'Hugging Face',
    description: 'Models served through the Hugging Face router.',
    category: 'Model APIs',
    baseUrl: 'https://router.huggingface.co/v1',
    apiFormat: 'openai',
    authMode: 'bearer',
    icon: lobeIcon('huggingface'),
    monogram: 'HF',
  },
  {
    id: 'sambanova',
    name: 'SambaNova',
    description: 'Hosted inference on SambaNova Cloud.',
    category: 'Model APIs',
    baseUrl: 'https://api.sambanova.ai/v1',
    apiFormat: 'openai',
    authMode: 'bearer',
    icon: lobeIcon('sambanova'),
    monogram: 'S',
  },
  {
    id: 'dashscope',
    name: 'DashScope',
    description: 'Qwen models through the compatible endpoint.',
    category: 'Model APIs',
    baseUrl: 'https://dashscope-intl.aliyuncs.com/compatible-mode/v1',
    apiFormat: 'openai',
    authMode: 'bearer',
    icon: lobeIcon('qwen'),
    monogram: 'Q',
  },
  {
    id: 'minimax',
    name: 'MiniMax',
    description: 'MiniMax language models and reasoning.',
    category: 'Model APIs',
    baseUrl: 'https://api.minimax.io/v1',
    apiFormat: 'openai',
    authMode: 'bearer',
    icon: lobeIcon('minimax'),
    monogram: 'M',
  },
  {
    id: 'moonshot',
    name: 'Moonshot AI',
    description: 'Kimi models through an OpenAI-style API.',
    category: 'Model APIs',
    baseUrl: 'https://api.moonshot.ai/v1',
    apiFormat: 'openai',
    authMode: 'bearer',
    icon: lobeMonoIcon('moonshot'),
    monogram: 'K',
  },
  {
    id: 'zai',
    name: 'Z.ai',
    description: 'GLM models from the Z.ai platform.',
    category: 'Model APIs',
    baseUrl: 'https://api.z.ai/api/paas/v4',
    apiFormat: 'openai',
    authMode: 'bearer',
    icon: lobeMonoIcon('zai'),
    monogram: 'Z',
  },
  {
    id: 'novita',
    name: 'Novita AI',
    description: 'Serverless access to open model families.',
    category: 'Model APIs',
    baseUrl: 'https://api.novita.ai/v3/openai',
    apiFormat: 'openai',
    authMode: 'bearer',
    icon: lobeIcon('novita'),
    monogram: 'N',
  },
  {
    id: 'nebius',
    name: 'Nebius AI Studio',
    description: 'Hosted open models from Nebius AI Studio.',
    category: 'Model APIs',
    baseUrl: 'https://api.studio.nebius.com/v1',
    apiFormat: 'openai',
    authMode: 'bearer',
    icon: lobeMonoIcon('nebius'),
    monogram: 'N',
  },
  {
    id: 'featherless',
    name: 'Featherless AI',
    description: 'On-demand serverless open-model inference.',
    category: 'Model APIs',
    baseUrl: 'https://api.featherless.ai/v1',
    apiFormat: 'openai',
    authMode: 'bearer',
    icon: lobeIcon('featherless'),
    monogram: 'F',
  },
  {
    id: 'friendli',
    name: 'FriendliAI',
    description: 'Friendli serverless model endpoints.',
    category: 'Model APIs',
    baseUrl: 'https://api.friendli.ai/serverless/v1',
    apiFormat: 'openai',
    authMode: 'bearer',
    icon: lobeMonoIcon('friendli'),
    monogram: 'F',
  },
  {
    id: 'vercel-ai-gateway',
    name: 'Vercel AI Gateway',
    description: 'A unified endpoint for hosted model providers.',
    category: 'Model APIs',
    baseUrl: 'https://ai-gateway.vercel.sh/v1',
    apiFormat: 'openai',
    authMode: 'bearer',
    icon: lobeMonoIcon('vercel'),
    monogram: 'V',
  },
  {
    id: 'cometapi',
    name: 'CometAPI',
    description: 'Unified hosted access to model APIs.',
    category: 'Model APIs',
    baseUrl: 'https://api.cometapi.com/v1',
    apiFormat: 'openai',
    authMode: 'bearer',
    icon: lobeIcon('cometapi'),
    monogram: 'C',
  },
  {
    id: 'sakana',
    name: 'Sakana AI',
    description: 'Models served by Sakana AI.',
    category: 'Model APIs',
    baseUrl: 'https://api.sakana.ai/v1',
    apiFormat: 'openai',
    authMode: 'bearer',
    icon: 'https://console.sakana.ai/icon.svg',
    monogram: 'S',
  },
  {
    id: 'ollama',
    name: 'Ollama',
    description: 'Connect a private Ollama endpoint.',
    category: 'Private runtimes',
    baseUrl: '',
    apiFormat: 'openai',
    authMode: 'none',
    icon: lobeMonoIcon('ollama'),
    monogram: 'O',
  },
  {
    id: 'lm-studio',
    name: 'LM Studio',
    description: 'Connect a private LM Studio endpoint.',
    category: 'Private runtimes',
    baseUrl: '',
    apiFormat: 'openai',
    authMode: 'none',
    icon: lobeMonoIcon('lmstudio'),
    monogram: 'LM',
  },
  {
    id: 'xinference',
    name: 'Xinference',
    description: 'Connect a private Xinference endpoint.',
    category: 'Private runtimes',
    baseUrl: '',
    apiFormat: 'openai',
    authMode: 'bearer',
    icon: lobeIcon('xinference'),
    monogram: 'X',
  },
  {
    id: 'nvidia-riva',
    name: 'NVIDIA Riva',
    description: 'Connect a private NVIDIA Riva endpoint.',
    category: 'Private runtimes',
    baseUrl: '',
    apiFormat: 'openai',
    authMode: 'bearer',
    icon: lobeIcon('nvidia'),
    monogram: 'N',
  },
  {
    id: 'triton',
    name: 'NVIDIA Triton',
    description: 'Connect a private NVIDIA Triton endpoint.',
    category: 'Private runtimes',
    baseUrl: '',
    apiFormat: 'openai',
    authMode: 'bearer',
    icon: lobeIcon('nvidia'),
    monogram: 'N',
  },
  {
    id: 'docker-model-runner',
    name: 'Docker Model Runner',
    description: 'Connect models running with Docker.',
    category: 'Private runtimes',
    baseUrl: '',
    apiFormat: 'openai',
    authMode: 'none',
    icon: 'https://www.docker.com/wp-content/uploads/2022/03/Moby-logo.png',
    monogram: 'D',
  },
  {
    id: 'lemonade',
    name: 'Lemonade',
    description: 'Connect a private Lemonade endpoint.',
    category: 'Private runtimes',
    baseUrl: '',
    apiFormat: 'openai',
    authMode: 'none',
    icon: '/amd.png',
    monogram: 'L',
  },
]

interface ProviderLookupInput {
  backendName?: string
  baseUrl?: string
  apiFormat?: string
}

function normalizedProviderID(value?: string): string {
  return (value ?? '')
    .trim()
    .toLowerCase()
    .replace(/-primary$/, '')
}

function providerHost(value?: string): string {
  if (!value) return ''
  try {
    return new URL(value).hostname.toLowerCase()
  } catch {
    return ''
  }
}

export function findModelProviderPreset({
  backendName,
  baseUrl,
  apiFormat,
}: ProviderLookupInput): ModelProviderPreset | undefined {
  const providerID = normalizedProviderID(backendName)
  const exact = modelProviderCatalog.find((provider) => provider.id === providerID)
  if (exact) return exact

  const host = providerHost(baseUrl)
  if (host) {
    const hostMatch = modelProviderCatalog.find(
      (provider) => providerHost(provider.baseUrl) === host,
    )
    if (hostMatch) return hostMatch
  }

  if (apiFormat === 'anthropic') {
    return modelProviderCatalog.find((provider) => provider.id === 'anthropic')
  }
  if (apiFormat === 'openai') {
    return modelProviderCatalog.find((provider) => provider.id === 'openai-compatible')
  }
  return undefined
}
