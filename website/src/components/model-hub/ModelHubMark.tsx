import React, { useEffect, useState } from 'react'
import useBaseUrl from '@docusaurus/useBaseUrl'
import Ai2 from '@lobehub/icons/es/Ai2/components/Color'
import Ai21 from '@lobehub/icons/es/Ai21/components/BrandColor'
import Bedrock from '@lobehub/icons/es/Bedrock/components/Color'
import Baidu from '@lobehub/icons/es/Baidu/components/Color'
import ByteDance from '@lobehub/icons/es/ByteDance/components/Color'
import Cerebras from '@lobehub/icons/es/Cerebras/components/Color'
import Claude from '@lobehub/icons/es/Claude/components/Color'
import Cohere from '@lobehub/icons/es/Cohere/components/Color'
import CometAPI from '@lobehub/icons/es/CometAPI/components/Color'
import DeepInfra from '@lobehub/icons/es/DeepInfra/components/Color'
import DeepSeek from '@lobehub/icons/es/DeepSeek/components/Color'
import Featherless from '@lobehub/icons/es/Featherless/components/Color'
import Fireworks from '@lobehub/icons/es/Fireworks/components/Color'
import Friendli from '@lobehub/icons/es/Friendli/components/Mono'
import Gemini from '@lobehub/icons/es/Gemini/components/Color'
import Grok from '@lobehub/icons/es/Grok/components/Mono'
import Groq from '@lobehub/icons/es/Groq/components/Mono'
import HuggingFace from '@lobehub/icons/es/HuggingFace/components/Color'
import InternLM from '@lobehub/icons/es/InternLM/components/Color'
import Kimi from '@lobehub/icons/es/Kimi/components/Color'
import LG from '@lobehub/icons/es/LG/components/Color'
import LmStudio from '@lobehub/icons/es/LmStudio/components/Mono'
import Meta from '@lobehub/icons/es/Meta/components/Color'
import Microsoft from '@lobehub/icons/es/Microsoft/components/Color'
import Minimax from '@lobehub/icons/es/Minimax/components/Color'
import Mistral from '@lobehub/icons/es/Mistral/components/Color'
import Nova from '@lobehub/icons/es/Nova/components/Color'
import Nebius from '@lobehub/icons/es/Nebius/components/Mono'
import Novita from '@lobehub/icons/es/Novita/components/Color'
import Nvidia from '@lobehub/icons/es/Nvidia/components/Color'
import Ollama from '@lobehub/icons/es/Ollama/components/Mono'
import OpenAI from '@lobehub/icons/es/OpenAI/components/Mono'
import OpenRouter from '@lobehub/icons/es/OpenRouter/components/Color'
import Perplexity from '@lobehub/icons/es/Perplexity/components/Color'
import Qwen from '@lobehub/icons/es/Qwen/components/Color'
import Snowflake from '@lobehub/icons/es/Snowflake/components/Color'
import SambaNova from '@lobehub/icons/es/SambaNova/components/Color'
import Stepfun from '@lobehub/icons/es/Stepfun/components/Mono'
import TII from '@lobehub/icons/es/TII/components/Color'
import Tencent from '@lobehub/icons/es/Tencent/components/Color'
import Together from '@lobehub/icons/es/Together/components/Color'
import Upstage from '@lobehub/icons/es/Upstage/components/Color'
import Vercel from '@lobehub/icons/es/Vercel/components/Mono'
import Vllm from '@lobehub/icons/es/Vllm/components/Color'
import XiaomiMiMo from '@lobehub/icons/es/XiaomiMiMo/components/Mono'
import Yi from '@lobehub/icons/es/Yi/components/Color'
import Xinference from '@lobehub/icons/es/Xinference/components/Color'
import Zhipu from '@lobehub/icons/es/Zhipu/components/Color'

import type { CatalogPresentation } from '../../data/modelHubCatalogTypes'
import styles from './modelHubShared.module.css'

const packageIcons: Record<string, typeof OpenAI> = {
  ai2: Ai2,
  ai21: Ai21,
  anthropic: Claude,
  baidu: Baidu,
  bedrock: Bedrock,
  bytedance: ByteDance,
  cerebras: Cerebras,
  cohere: Cohere,
  cometapi: CometAPI,
  deepinfra: DeepInfra,
  deepseek: DeepSeek,
  featherless: Featherless,
  fireworks: Fireworks,
  friendli: Friendli,
  gemini: Gemini,
  google: Gemini,
  groq: Groq,
  huggingface: HuggingFace,
  internlm: InternLM,
  exaone: LG,
  lg: LG,
  lmstudio: LmStudio,
  meta: Meta,
  microsoft: Microsoft,
  minimax: Minimax,
  mistral: Mistral,
  moonshot: Kimi,
  nebius: Nebius,
  novita: Novita,
  nova: Nova,
  nvidia: Nvidia,
  ollama: Ollama,
  openai: OpenAI,
  openrouter: OpenRouter,
  perplexity: Perplexity,
  qwen: Qwen,
  sambanova: SambaNova,
  snowflake: Snowflake,
  stepfun: Stepfun,
  tencent: Tencent,
  together: Together,
  tii: TII,
  upstage: Upstage,
  vercel: Vercel,
  vllm: Vllm,
  xai: Grok,
  xiaomimimo: XiaomiMiMo,
  xinference: Xinference,
  yi: Yi,
  zai: Zhipu,
}

export function CatalogMark({
  presentation,
  large = false,
}: {
  presentation: CatalogPresentation
  large?: boolean
}) {
  const logo = presentation.logo ?? ''
  const packageID = logo.startsWith('package:') ? logo.slice('package:'.length) : ''
  const publicLogo = logo.startsWith('public:') ? logo.slice('public:'.length) : ''
  const resolvedPublicLogo = useBaseUrl(publicLogo)
  const directLogo = publicLogo
    ? resolvedPublicLogo
    : logo.startsWith('url:')
      ? logo.slice('url:'.length)
      : ''
  const [logoFailed, setLogoFailed] = useState(false)
  useEffect(() => setLogoFailed(false), [directLogo])
  const Icon = packageIcons[packageID]

  return (
    <span
      className={`${styles.catalogMark} ${large ? styles.catalogMarkLarge : ''}`}
      aria-hidden="true"
    >
      {Icon
        ? (
            <Icon size={large ? 29 : 21} />
          )
        : directLogo && !logoFailed
          ? (
              <img src={directLogo} alt="" onError={() => setLogoFailed(true)} />
            )
          : (
              presentation.monogram
            )}
    </span>
  )
}
