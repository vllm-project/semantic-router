import anthropic from '@lobehub/icons-static-svg/icons/anthropic.svg'
import ai2 from '@lobehub/icons-static-svg/icons/ai2-color.svg'
import ai21 from '@lobehub/icons-static-svg/icons/ai21-brand-color.svg'
import cerebras from '@lobehub/icons-static-svg/icons/cerebras-color.svg'
import cohere from '@lobehub/icons-static-svg/icons/cohere-color.svg'
import cometapi from '@lobehub/icons-static-svg/icons/cometapi-color.svg'
import deepinfra from '@lobehub/icons-static-svg/icons/deepinfra-color.svg'
import deepseek from '@lobehub/icons-static-svg/icons/deepseek-color.svg'
import featherless from '@lobehub/icons-static-svg/icons/featherless-color.svg'
import fireworks from '@lobehub/icons-static-svg/icons/fireworks-color.svg'
import friendli from '@lobehub/icons-static-svg/icons/friendli.svg'
import gemini from '@lobehub/icons-static-svg/icons/gemini-color.svg'
import groq from '@lobehub/icons-static-svg/icons/groq.svg'
import huggingface from '@lobehub/icons-static-svg/icons/huggingface-color.svg'
import internlm from '@lobehub/icons-static-svg/icons/internlm-color.svg'
import lmstudio from '@lobehub/icons-static-svg/icons/lmstudio.svg'
import meta from '@lobehub/icons-static-svg/icons/meta-color.svg'
import minimax from '@lobehub/icons-static-svg/icons/minimax-color.svg'
import mistral from '@lobehub/icons-static-svg/icons/mistral-color.svg'
import moonshot from '@lobehub/icons-static-svg/icons/moonshot.svg'
import nebius from '@lobehub/icons-static-svg/icons/nebius.svg'
import novita from '@lobehub/icons-static-svg/icons/novita-color.svg'
import nvidia from '@lobehub/icons-static-svg/icons/nvidia-color.svg'
import ollama from '@lobehub/icons-static-svg/icons/ollama.svg'
import openai from '@lobehub/icons-static-svg/icons/openai.svg'
import openrouter from '@lobehub/icons-static-svg/icons/openrouter-color.svg'
import perplexity from '@lobehub/icons-static-svg/icons/perplexity-color.svg'
import qwen from '@lobehub/icons-static-svg/icons/qwen-color.svg'
import sambanova from '@lobehub/icons-static-svg/icons/sambanova-color.svg'
import snowflake from '@lobehub/icons-static-svg/icons/snowflake-color.svg'
import stepfun from '@lobehub/icons-static-svg/icons/stepfun-color.svg'
import together from '@lobehub/icons-static-svg/icons/together-color.svg'
import tii from '@lobehub/icons-static-svg/icons/tii-color.svg'
import vercel from '@lobehub/icons-static-svg/icons/vercel.svg'
import vllm from '@lobehub/icons-static-svg/icons/vllm-color.svg'
import xai from '@lobehub/icons-static-svg/icons/xai.svg'
import xinference from '@lobehub/icons-static-svg/icons/xinference-color.svg'
import yi from '@lobehub/icons-static-svg/icons/yi-color.svg'
import zai from '@lobehub/icons-static-svg/icons/zai.svg'

export const modelProviderIconAssets: Record<string, string> = {
  ai2,
  ai21,
  anthropic,
  cerebras,
  cohere,
  cometapi,
  deepinfra,
  deepseek,
  featherless,
  fireworks,
  friendli,
  gemini,
  groq,
  huggingface,
  internlm,
  lmstudio,
  meta,
  minimax,
  mistral,
  moonshot,
  nebius,
  novita,
  nvidia,
  ollama,
  openai,
  openrouter,
  perplexity,
  qwen,
  sambanova,
  snowflake,
  stepfun,
  together,
  tii,
  vercel,
  vllm,
  xai,
  xinference,
  yi,
  zai,
}

export const monochromeModelProviderIcons = new Set([
  anthropic,
  friendli,
  groq,
  lmstudio,
  moonshot,
  nebius,
  ollama,
  openai,
  vercel,
  xai,
  zai,
])

export const resolveModelCatalogIcon = (source: string): string => {
  if (source.startsWith('package:')) {
    return modelProviderIconAssets[source.slice('package:'.length)] ?? ''
  }
  if (source.startsWith('public:')) return source.slice('public:'.length)
  if (source.startsWith('url:')) return source.slice('url:'.length)
  return ''
}
