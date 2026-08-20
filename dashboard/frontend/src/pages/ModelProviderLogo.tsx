import { useEffect, useState, type CSSProperties } from 'react'

import { getModelProvider } from './modelProviderCatalog'
import styles from './ModelProviderLogo.module.css'

interface ModelProviderLogoProps {
  provider?: string
  size?: 'small' | 'medium' | 'large'
  className?: string
}

const LOBE_ICON_VERSION = '1.90.0'
const LOBE_ICON_BASE = `https://unpkg.com/@lobehub/icons-static-svg@${LOBE_ICON_VERSION}/icons`

const lobeIcons: Record<string, { slug: string; color?: boolean }> = {
  vllm: { slug: 'vllm', color: true },
  'openai-compatible': { slug: 'openai' },
  openrouter: { slug: 'openrouter' },
  openai: { slug: 'openai' },
  anthropic: { slug: 'anthropic' },
  'google-ai-studio': { slug: 'gemini', color: true },
  'azure-openai': { slug: 'azure', color: true },
  'azure-ai-foundry': { slug: 'azureai', color: true },
  'amazon-bedrock': { slug: 'bedrock', color: true },
  'vertex-ai': { slug: 'vertexai', color: true },
  deepseek: { slug: 'deepseek', color: true },
  mistral: { slug: 'mistral', color: true },
  groq: { slug: 'groq' },
  'together-ai': { slug: 'together', color: true },
  'fireworks-ai': { slug: 'fireworks', color: true },
  cerebras: { slug: 'cerebras', color: true },
  xai: { slug: 'xai' },
  perplexity: { slug: 'perplexity', color: true },
  cohere: { slug: 'cohere', color: true },
  ai21: { slug: 'ai21-brand', color: true },
  deepinfra: { slug: 'deepinfra', color: true },
  'hugging-face': { slug: 'huggingface', color: true },
  replicate: { slug: 'replicate-brand' },
  cloudflare: { slug: 'cloudflare', color: true },
  'nvidia-nim': { slug: 'nvidia', color: true },
  sambanova: { slug: 'sambanova', color: true },
  snowflake: { slug: 'snowflake', color: true },
  watsonx: { slug: 'ibm' },
  volcengine: { slug: 'volcengine', color: true },
  dashscope: { slug: 'qwen', color: true },
  minimax: { slug: 'minimax', color: true },
  moonshot: { slug: 'moonshot' },
  zai: { slug: 'zai' },
  novita: { slug: 'novita', color: true },
  baseten: { slug: 'baseten' },
  nebius: { slug: 'nebius' },
  hyperbolic: { slug: 'hyperbolic', color: true },
  friendli: { slug: 'friendli' },
  lambda: { slug: 'lambda' },
  'vercel-ai-gateway': { slug: 'vercel' },
  cometapi: { slug: 'cometapi', color: true },
  heroku: { slug: 'heroku', color: true },
  clarifai: { slug: 'clarifai', color: true },
  'github-models': { slug: 'github' },
  'github-copilot': { slug: 'githubcopilot' },
  cursor: { slug: 'cursor' },
  ollama: { slug: 'ollama' },
  'lm-studio': { slug: 'lmstudio' },
  xinference: { slug: 'xinference', color: true },
}

export default function ModelProviderLogo({
  provider,
  size = 'medium',
  className = '',
}: ModelProviderLogoProps) {
  const definition = getModelProvider(provider)
  const [imageFailed, setImageFailed] = useState(false)
  const icon = lobeIcons[definition.id]
  useEffect(() => setImageFailed(false), [definition.id])
  const style = { '--provider-accent': definition.accent } as CSSProperties
  return (
    <span
      className={`${styles.logo} ${styles[size]} ${className}`}
      style={style}
      title={definition.name}
      aria-label={`${definition.name} logo`}
    >
      {definition.id === 'amd-atom' ? (
        <img src="/amd.png" alt="" />
      ) : icon && !imageFailed ? (
        <img
          src={`${LOBE_ICON_BASE}/${icon.slug}${icon.color ? '-color' : ''}.svg`}
          alt=""
          className={icon.color ? styles.colorIcon : styles.monoIcon}
          onError={() => setImageFailed(true)}
        />
      ) : (
        <span>{definition.shortName}</span>
      )}
    </span>
  )
}
