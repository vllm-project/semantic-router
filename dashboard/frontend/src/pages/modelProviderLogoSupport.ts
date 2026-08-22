const LOBE_ICON_VERSION = '1.90.0'
const LOBE_ICON_BASE = `https://unpkg.com/@lobehub/icons-static-svg@${LOBE_ICON_VERSION}/icons`

export const providerLobeIcons: Record<string, { slug: string; color?: boolean }> = {
  vllm: { slug: 'vllm', color: true },
  'openai-compatible': { slug: 'openai' },
  openrouter: { slug: 'openrouter' },
  openai: { slug: 'openai' },
  anthropic: { slug: 'anthropic' },
  'google-ai-studio': { slug: 'gemini', color: true },
  deepseek: { slug: 'deepseek', color: true },
  mistral: { slug: 'mistral', color: true },
  groq: { slug: 'groq' },
  'together-ai': { slug: 'together', color: true },
  'fireworks-ai': { slug: 'fireworks', color: true },
  cerebras: { slug: 'cerebras', color: true },
  xai: { slug: 'xai' },
  perplexity: { slug: 'perplexity', color: true },
  cohere: { slug: 'cohere', color: true },
  deepinfra: { slug: 'deepinfra', color: true },
  'hugging-face': { slug: 'huggingface', color: true },
  'nvidia-nim': { slug: 'nvidia', color: true },
  sambanova: { slug: 'sambanova', color: true },
  dashscope: { slug: 'qwen', color: true },
  minimax: { slug: 'minimax', color: true },
  moonshot: { slug: 'moonshot' },
  zai: { slug: 'zai' },
  novita: { slug: 'novita', color: true },
  nebius: { slug: 'nebius' },
  featherless: { slug: 'featherless', color: true },
  friendli: { slug: 'friendli' },
  'vercel-ai-gateway': { slug: 'vercel' },
  cometapi: { slug: 'cometapi', color: true },
  ollama: { slug: 'ollama' },
  'lm-studio': { slug: 'lmstudio' },
  xinference: { slug: 'xinference', color: true },
  'nvidia-riva': { slug: 'nvidia', color: true },
  triton: { slug: 'nvidia', color: true },
}

export const providerDirectIcons: Record<string, string> = {
  'amd-atom': '/amd.png',
  sglang: 'https://raw.githubusercontent.com/sgl-project/sgl-docs/main/favicon.png',
  sakana: 'https://console.sakana.ai/icon.svg',
  'docker-model-runner': 'https://www.docker.com/wp-content/uploads/2022/03/Moby-logo.png',
  lemonade: '/amd.png',
}

export const getModelProviderLogoSource = (providerId: string) => {
  if (providerDirectIcons[providerId]) return providerDirectIcons[providerId]
  const icon = providerLobeIcons[providerId]
  if (!icon) return ''
  return `${LOBE_ICON_BASE}/${icon.slug}${icon.color ? '-color' : ''}.svg`
}
