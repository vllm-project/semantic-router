import React from 'react'
import useBaseUrl from '@docusaurus/useBaseUrl'

const ink = '#102B40'
const blue = '#008BE8'
const amber = '#E8A000'
const muted = '#526D80'

function Provider({ x, y, asset, name, wide = false, secondLine, logoWidth, logoHeight = 94 }: {
  x: number
  y: number
  asset: string
  name?: string
  wide?: boolean
  secondLine?: string
  logoWidth?: number
  logoHeight?: number
}) {
  const source = asset.startsWith('/') ? asset : `/img/blog/system-one-auto/providers/${asset}`
  const width = logoWidth ?? (wide ? 192 : 94)
  return (
    <g>
      <image href={useBaseUrl(source)} x={x - width / 2} y={y} width={width} height={logoHeight} preserveAspectRatio="xMidYMid meet" />
      {name && <text x={x} y={y + 140} textAnchor="middle" fontSize="32" fontWeight="650" fill={ink}>{name}</text>}
      {secondLine && <text x={x} y={y + 180} textAnchor="middle" fontSize="32" fill={muted}>{secondLine}</text>}
    </g>
  )
}

export function ModelEcosystemsFigure() {
  return (
    <svg width={2000} height={1080} viewBox="0 0 2000 1080" role="img" aria-label="Language model APIs and the native System One API share vLLM Semantic Router, serving separate open and hosted model ecosystems." xmlns="http://www.w3.org/2000/svg" fontFamily="Inter, Arial, sans-serif">
      <title>Route language. Now, route decisions.</title>
      <desc>Chat Completions, Responses, and Messages serve language models. The native System One API serves decision models. Both share the vLLM Semantic Router frontend. Provider logos illustrate open and hosted ecosystems, not tested integrations.</desc>
      <rect width="2000" height="1080" fill="#FFFFFF" />
      <rect x="1090" y="222" width="400" height="295" rx="24" fill="#EFF8FF" />
      <rect x="1550" y="222" width="380" height="295" rx="24" fill="#FFF7E6" />
      <rect x="1090" y="556" width="500" height="464" rx="24" fill="#EFF8FF" />
      <rect x="1650" y="556" width="280" height="464" rx="24" fill="#FFF7E6" />

      <text x="70" y="130" fontSize="64" fontWeight="800" letterSpacing="-1.6" fill={ink}>Route language.</text>
      <text x="683" y="130" fontSize="64" fontWeight="800" letterSpacing="-1.6" fill={blue}>Now, route decisions.</text>

      <text x="70" y="281" fontSize="42" fontWeight="750" fill={ink}>Language APIs</text>
      <g fontFamily="'IBM Plex Mono', Consolas, monospace" fontSize="32" fill={muted}>
        <text x="70" y="344">/v1/chat/completions</text>
        <text x="70" y="394">/v1/responses</text>
        <text x="70" y="444">/v1/messages</text>
      </g>
      <text x="70" y="737" fontSize="42" fontWeight="750" fill={ink}>System One API</text>
      <text x="70" y="800" fontFamily="'IBM Plex Mono', Consolas, monospace" fontSize="34" fill={muted}>/v1/systemone</text>

      <g fill="none" strokeWidth="4" strokeLinecap="round" strokeLinejoin="round">
        <path d="M453 375h35q22 0 22 22v99q0 22 22 22h26m-11-11 11 11-11 11" stroke={blue} />
        <path d="M453 762h35q22 0 22-22V584q0-22 22-22h26m-11-11 11 11-11 11" stroke={amber} />
        <path d="M900 518h20q22 0 22-22V390q0-22 22-22h106m-11-11 11 11-11 11" stroke={blue} />
        <path d="M900 562h20q22 0 22 22v92q0 22 22 22h106m-11-11 11 11-11 11" stroke={amber} />
      </g>
      <image href={useBaseUrl('/img/vllm-sr-logo.light.png')} x="580" y="470" width="300" height="98" preserveAspectRatio="xMidYMid meet" />
      <text x="730" y="628" textAnchor="middle" fontSize="32" fontWeight="650" fill={ink}>One serving layer</text>

      <g fontSize="32" fontWeight="650">
        <text x="1126" y="277" fill={blue}>Open</text>
        <text x="1586" y="277" fill="#9E6900">Close</text>
        <text x="1126" y="613" fill={blue}>Open</text>
        <text x="1686" y="613" fill="#9E6900">Close</text>
      </g>
      <Provider x={1180} y={319} asset="qwen.svg" name="Qwen" />
      <Provider x={1380} y={319} asset="deepseek.svg" name="DeepSeek" />
      <Provider x={1645} y={319} asset="openai.svg" name="OpenAI" />
      <Provider x={1835} y={319} asset="anthropic.svg" name="Anthropic" />

      <Provider x={1220} y={648} asset="/img/vllm-sr-logo.light.png" name="Decision 2.0" wide />
      <Provider x={1460} y={648} asset="cloudflare.png" name="Clef" />
      <Provider x={1220} y={860} asset="perplexity.png" name="Decider" />
      <Provider x={1450} y={842} asset="laya.png" logoWidth={250} logoHeight={130} />
      <Provider x={1790} y={648} asset="fastino.png" name="GLiDE" secondLine="no-thinking" />
      <Provider x={1790} y={860} asset="typesafe.png" name="Jev" />
    </svg>
  )
}
